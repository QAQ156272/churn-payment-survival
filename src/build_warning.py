"""Generate the train-only scorer and the historical Phase 3 handoff."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

if __package__:
    from . import evaluate_cox, train_cox, warning
else:
    import evaluate_cox
    import train_cox
    import warning


def build_model(train, features):
    model, issues = train_cox.fit(train, features, penalizer=0)
    if issues:
        raise ValueError("; ".join(issues))
    reference = train[features].median()
    baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
    hazard = baseline.reindex(range(1, warning.LAST_DAY + 1), method="ffill").fillna(0)
    relative = model.predict_partial_hazard(reference.to_frame().T).iloc[0]
    risk = evaluate_cox.survival_at_days(model, train[features])[warning.DAY]
    spec = {
        "model": "cox_b_next_purchase_v1",
        "features": features,
        "reference": reference.to_dict(),
        "coef": model.params_.to_dict(),
        "reference_cumulative_hazard": (hazard * relative).tolist(),
        "thresholds": risk.quantile([0.25, 0.5, 0.75]).tolist(),
    }
    if len(set(spec["thresholds"])) != 3:
        raise ValueError("Training quantiles do not define four distinct tiers")
    return spec


def tier_summary(test, scored):
    cohort = evaluate_cox.horizon_cohort(test, warning.DAY)
    matched = scored.set_index("user_id").loc[cohort["user_id"].astype(str)].reset_index(drop=True)
    matched["no_purchase"] = ~cohort["purchased"].to_numpy()
    groups = pd.DataFrame(index=pd.Index(warning.TIERS[::-1], name="risk_tier"))
    groups["scored_n"] = scored.groupby("risk_tier").size()
    groups["evaluable_n"] = matched.groupby("risk_tier").size()
    groups["no_purchase_n"] = matched.groupby("risk_tier")["no_purchase"].sum()
    groups = groups.fillna(0).astype(int)
    groups["mean_predicted"] = matched.groupby("risk_tier")["risk_score"].mean()
    groups["observed_rate"] = groups["no_purchase_n"] / groups["evaluable_n"].replace(0, np.nan)
    groups["event_share"] = groups["no_purchase_n"] / groups["no_purchase_n"].sum()
    low, high = proportion_confint(groups["no_purchase_n"], groups["evaluable_n"].replace(0, np.nan),
                                    method="wilson")
    return groups.assign(ci_low=low, ci_high=high).reset_index()


def report(spec, summary, candidates):
    rates = summary["observed_rate"]
    strict = rates.notna().all() and rates.diff().iloc[1:].lt(0).all()
    p25, p50, p75 = spec["thresholds"]
    lines = [
        "# 运营交付：30 天未复购分层与预警", "",
        "**用途：**排序与试验入组。「高风险」是打分后 30 天内更可能没有下一单，不代表永久流失。"
        "分数是未校准的原 Cox B 模型值，不能直接当成订单或挽回量预测。", "",
        "## 分层与单调性", "",
        "Cox B 仅用原训练集、六个打分前特征拟合；以 S(30) 为分数，越高越靠前。"
        f"训练集 P25/P50/P75 为 {p25:.6f} / {p50:.6f} / {p75:.6f}；"
        "从低到高四档，等于切点归入较低一档。测试集和上传表沿用同一套切点，不重新分位。", "",
        "| 档位 | 打分人数 | 满 30 天 | 未购买人数 | 平均分 | 实际未购买率（95% CI） | 占未购买总人数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        observed = (f"{r.observed_rate:.1%}（{r.ci_low:.1%}–{r.ci_high:.1%}）"
                    if r.evaluable_n else "不可评估")
        lines.append(f"| {r.risk_tier} | {r.scored_n} | {r.evaluable_n} | {r.no_purchase_n} | "
                     f"{r.mean_predicted:.1%} | {observed} | {r.event_share:.1%} |")
    lines += [
        "",
        f"**严格单调性：{'通过' if strict else '未通过'}。**按高到低看，"
        "逐档用实际未购买率点估计验证；Wilson 区间仅描述各档不确定性，不检验档间差异。"
        "只用截至标签截止日有完整 30 天观察窗口的用户计算实际比例，短随访不记负例。"
        "未购买人数占比的分母是完整窗口中全部未购买用户。", "",
    ]
    if strict:
        high, mid_high, mid, low = list(summary.itertuples(index=False))
        lines += [
            f"高档覆盖了 {high.event_share:.1%} 的未购买用户，与低档相差 "
            f"{(high.observed_rate - low.observed_rate) * 100:.1f} 个百分点；高档优先用于试验招募。",
            f"中高和中仅差 {(mid_high.observed_rate - mid.observed_rate) * 100:.1f} 个百分点，区间重叠；"
            "在新时间段验证前不要给这两档不同的处置成本。", "",
        ]
    if not strict:
        for i in range(1, len(summary)):
            upper, lower = summary.iloc[i - 1], summary.iloc[i]
            if pd.isna(lower.observed_rate) or pd.isna(upper.observed_rate):
                lines.append(f"- {upper.risk_tier} → {lower.risk_tier}：窗口覆盖不足，无法比较。")
            elif lower.observed_rate >= upper.observed_rate:
                lines.append(f"- {upper.risk_tier} → {lower.risk_tier}：未购买率 "
                             f"{upper.observed_rate:.1%} → {lower.observed_rate:.1%}，未下降。"
                             f"两档可评估样本 {upper.evaluable_n}/{lower.evaluable_n}，"
                             f"预测均值 {upper.mean_predicted:.1%}/{lower.mean_predicted:.1%}；"
                             "没有用测试结果重切档，不能把这两档当作已验证的不同处置等级。")
        lines += ["", "回查优先看倒序档在不同打分月份是否反复出现，以及各月份样本构成、"
                  "近末购天数和首单等特征是否漂移；用新时间段确认后才调整分层或阈值。", ""]
    lines += [
        "## 预警名单和影响面", "",
        f"历史测试集输出 {len(candidates)} 位高档候选，按原始分数降序、同分按用户 ID 排序，"
        "上限 1,000 人，不拿其他档补齐。名单只回放当时打分，不是正在流失的客户清单。"
        "字段含原始用户 ID、打分日、分数、档位、主要因素和动作；历史名单不提交到公开仓库。", "",
        "| 候选动作 | 人数 |", "|---|---:|",
    ]
    for action, n in candidates["suggested_action"].value_counts().items():
        lines.append(f"| {action} | {n} |")
    lines += [
        "", "### 如何处置", "",
        "- 距末购天数或发票数贡献居首：在打分后安排一次补货提醒或复购入口；发送前重查是否已购买。",
        "- 首单或窗口消费金额贡献居首：作为小额复购券的随机试验候选；按净毛利和券成本限制资格，默认不发券。",
        "- 商品种类或主购商品集中度贡献居首：有可用商品记录才推相关或常购商品；缺货或无权限则不发。",
        "- 因素是相对训练中位数对模型对数购买风险的贡献，不是流失原因。"
        "高风险不等于干预有效；人工召回仅在独立确认高价值、联系方式和合规许可后考虑。", "",
        "人数是可筛查的上限，不是预计触达或挽回人数。接入运营系统后先剔除无许可、已复购、"
        "重复触达、退订和不可用渠道。举例：每 100 位随机入组用户，若 30 天复购率净增 3 个百分点，"
        "相当于每 100 位约多 3 人购买；这是情景算术，不是模型估计。", "",
        "## A/B 试验", "",
        "1. 在打分后第 7 天，筛出高档且当时还没买、允许触达并未进入冲突活动的用户。"
        "按用户 ID 固定分配，按候选动作分层 1:1 随机；同一用户只入组一次。",
        "2. 对照组照常运营，处理组额外收到一项动作。先试低成本提醒；券和人工触达另设试验，"
        "不把成本差异混成一个结论。",
        "3. 主指标是随机后到原打分第 30 天结束的复购率，按意向治疗比较百分点差和 95% CI；"
        "护栏看增量净毛利、退订、投诉和券成本。停止规则、最小有意义增量、样本量和观察期在试验前登记。",
        "4. 历史名单的规模不是试验所需样本量。未达到预定样本时不因短期波动判断有效；"
        "试验期间持续收集，不在每次查看结果后提前宣布成功。", "",
        "## 限制", "",
        "历史样本是旧时期零售数据中的有 ID 首购客户。没有首购的游客、无有效交易者、"
        "仅有退货／取消记录者或未满 14 天特征窗的新用户不可直接打分。"
        "其他地区、渠道、价格体系、营销政策或更晚时期需重新验证区分度与校准。"
        "阶段二测试集已经用于评估，本阶段仍是同一历史留出集的回测，不是新的独立验证。",
    ]
    return "\n".join(lines) + "\n"


def main():
    frame, features = train_cox.load()
    if problems := train_cox.validate(frame, features):
        raise ValueError("; ".join(problems))
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    spec = build_model(train, features)
    scored, _ = warning.score(test, spec)
    summary = tier_summary(test, scored)
    dates = test.assign(user_id=test["user_id"].astype(str))[["user_id", "score_day"]]
    candidates = warning.warning_list(scored).merge(dates, on="user_id", validate="one_to_one")
    Path("models").mkdir(exist_ok=True)
    Path("models/cox_warning.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    summary.to_csv("reports/risk_tiers.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv("reports/warning_list.csv", index=False, encoding="utf-8-sig")
    Path("reports/operations_playbook.md").write_text(report(spec, summary, candidates), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Historical warning rows: {len(candidates)}")


if __name__ == "__main__":
    main()
