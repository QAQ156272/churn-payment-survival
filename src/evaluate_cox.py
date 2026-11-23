"""Phase 2 step 4: held-out concordance, calibration and a binary baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from xgboost import XGBClassifier, __version__ as xgb_version

if __package__:
    from . import train_cox as tc
else:
    import train_cox as tc

MAIN_DAY = 30
DAYS = (14, MAIN_DAY, 60)
COX = "Cox B"
XGB = "XGBoost"
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1,
)


def horizon_cohort(frame, day):
    # Membership depends on the scheduled follow-up, never on early purchases.
    scheduled = (frame["label_deadline"] - frame["score_day"]).dt.days + 1
    cohort = frame.loc[scheduled.ge(day)].copy()
    cohort["purchased"] = cohort["E"].eq(1) & cohort["T"].le(day)
    return cohort


def survival_at_days(model, predictors):
    # Evaluate the step function with events on day t included in 1-S(t).
    hazard = model.baseline_cumulative_hazard_.iloc[:, 0].reindex(DAYS, method="ffill").fillna(0)
    partial = model.predict_partial_hazard(predictors).reindex(predictors.index)
    result = pd.DataFrame(np.exp(-np.outer(partial, hazard)), index=predictors.index, columns=DAYS)
    if not np.isfinite(result).all().all() or result.diff(axis=1).iloc[:, 1:].gt(0).any().any():
        raise ValueError("Invalid Cox survival predictions")
    return result


def calibration(cohort, survival, model, day):
    aligned = survival.reindex(cohort.index)
    if not aligned.between(0, 1).all() or cohort.empty:
        raise ValueError("Empty cohort or invalid survival probabilities")
    groups = pd.qcut(aligned, q=5, labels=False, duplicates="drop").fillna(0).astype(int) + 1
    table = pd.DataFrame({
        "bin": groups, "survival": aligned, "not_purchased": ~cohort["purchased"],
    }).groupby("bin", as_index=False).agg(
        n=("survival", "size"), predicted_survival=("survival", "mean"),
        non_events=("not_purchased", "sum"),
    )
    table["observed_survival"] = table["non_events"] / table["n"]
    low, high = proportion_confint(table["non_events"], table["n"], method="wilson")
    return table.assign(model=model, day=day, ci_low=low, ci_high=high)


def fixed_scores(cohort, probabilities):
    p = probabilities.reindex(cohort.index)
    y = cohort["purchased"]
    if not p.between(0, 1).all() or y.nunique() != 2:
        raise ValueError("Invalid predictions or missing outcome class")
    return {
        "C@30": concordance_index(cohort["T"].clip(upper=MAIN_DAY), -p, y),
        "ROC AUC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "predicted_rate": p.mean(),
    }


def evaluate(train, test, feats):
    c_full = {}
    models = {}
    for name, columns in (("Cox A", [tc.BASELINE]), (COX, feats)):
        model, issues = tc.fit(train, columns, penalizer=0)
        if issues:
            raise ValueError(f"{name}: " + "; ".join(issues))
        models[name] = model
        c_full[name] = concordance_index(test["T"], -model.predict_partial_hazard(test[columns]), test["E"])

    train30 = horizon_cohort(train, MAIN_DAY)
    test30 = horizon_cohort(test, MAIN_DAY)
    classifier = XGBClassifier(**XGB_PARAMS)
    classifier.fit(train30[feats], train30["purchased"].astype(int))
    xgb_p = pd.Series(classifier.predict_proba(test30[feats])[:, 1], index=test30.index)
    survival = survival_at_days(models[COX], test[feats])
    metrics = {COX: fixed_scores(test30, 1 - survival[MAIN_DAY]), XGB: fixed_scores(test30, xgb_p)}

    bins, coverage = [], []
    for day in DAYS:
        cohort = horizon_cohort(test, day)
        bins.append(calibration(cohort, survival[day], COX, day))
        coverage.append((day, len(cohort), int(cohort["purchased"].sum()), survival.loc[cohort.index, day].mean()))
    bins.append(calibration(test30, 1 - xgb_p, XGB, MAIN_DAY))
    return c_full, metrics, pd.concat(bins, ignore_index=True), coverage, train30


def plot_calibration(bins, coverage, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.5), layout="constrained")
    colors = {COX: "#156a80", XGB: "#ac4e20"}
    fig.suptitle("Next purchase | Test-set calibration", x=0.06, ha="left", fontsize=16, weight="bold")
    for ax, (day, n, events, _) in zip(axes, coverage):
        ax.plot([0, 1], [0, 1], "--", color="#7c8791", linewidth=1, label="Perfect calibration")
        for name, group in bins[bins["day"].eq(day)].groupby("model", sort=False):
            obs = group["observed_survival"]
            ax.errorbar(group["predicted_survival"], obs,
                        yerr=[(obs - group["ci_low"]).clip(lower=0), (group["ci_high"] - obs).clip(lower=0)],
                        marker="o", markersize=4.5, linestyle="-", capsize=3,
                        linewidth=1.35, color=colors[name], label=name)
        ax.set_title(f"Day {day}  |  n={n}, purchases={events}", fontsize=10.5)
        ax.set(xlim=(0, 1.025), ylim=(0, 1.025), aspect="equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(xmax=1))
            axis.set_ticks(np.linspace(0, 1, 5))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.supxlabel("Mean predicted probability of no purchase by day t", fontsize=10)
    fig.supylabel("Observed probability of no purchase", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(test, c_full, metrics, coverage, train30, feats):
    cox, xgb = metrics[COX], metrics[XGB]
    _, n30, events30, _ = next(point for point in coverage if point[0] == MAIN_DAY)
    lines = [
        "# 第二阶段第 4 步：C-index、校准与二分类基线", "",
        "目标仍是打分后到下一笔有效购买的时间，不是流失；打分当天计第 1 天。"
        "采用第 3 步保留的原 Cox B，连同单特征 Cox A 在训练集无惩罚复拟合。"
        "XGBoost 使用固定配置；验证集和测试集不参与拟合、选特征、调参或校准。", "",
        f"**摘要：** Cox B 完整测试集 C-index 为 {c_full[COX]:.4f}。"
        f"在共同的 30 天队列上，Cox B / XGBoost 的 ROC AUC 分别为 {cox['ROC AUC']:.4f} / "
        f"{xgb['ROC AUC']:.4f}，平均预测购买率分别为 {cox['predicted_rate']:.1%} / "
        f"{xgb['predicted_rate']:.1%}，实际为 {events30 / n30:.1%}。"
        "排序和概率误差需要分别判断。", "",
        "## 完整测试集：Cox 排序能力", "",
        f"共 {len(test)} 人、{int(test['E'].sum())} 个事件、{int(test['E'].eq(0).sum())} 个右删失。"
        "用部分风险和原始 T/E 计算 Harrell C-index。该值反映可比较对的排序，受删失模式影响。", "",
        "| 模型 | Harrell C-index |", "|---|---:|",
        f"| Cox A（购后间隔） | {c_full['Cox A']:.4f} |",
        f"| Cox B（六特征） | {c_full[COX]:.4f} |", "",
        "## 同口径对照：30 天内是否购买", "",
        "只纳入打分日起到标签截止日有至少 30 个自然日的用户。是否入组只看日期，不看是否提前购买。"
        "第 30 天购买算正例，更晚购买或完整观察 30 天仍未购买算负例。"
        "测试时，两模型在同一批用户上使用同样的六个打分前特征。"
        "Cox 保留完整训练集，分类器仅使用完整窗口，结果是两套建模流程的对照，"
        "不能把差异单独归因于算法。", "",
        f"- XGBoost 二分类训练：{len(train30)} 人，{int(train30['purchased'].sum())} 个正例。",
        f"- XGBoost CPU {xgb_version}：" + ", ".join(f"`{k}={v}`" for k, v in XGB_PARAMS.items()) + "。",
        "- 六特征：" + ", ".join(feats) + "。", "",
        "| 模型 | C@30 ↑ | ROC AUC ↑ | AP ↑ | Brier ↓ | 平均预测购买率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['C@30']:.4f} | {row['ROC AUC']:.4f} | "
                     f"{row['AP']:.4f} | {row['Brier']:.4f} | {row['predicted_rate']:.1%} |")
    lines += [
        "",
        "C@30 在第 30 天截断随访，按两模型各自的 30 天购买概率给用户排序。"
        "它还比较窗口内谁更早，ROC AUC 则只区分正负例。"
        "完整随访 C-index 与 C@30 不可直接比较；XGBoost 没有输出完整生存曲线。", "",
        f"同一批测试用户上，XGBoost − Cox B：C@30 {xgb['C@30'] - cox['C@30']:+.4f}，"
        f"ROC AUC {xgb['ROC AUC'] - cox['ROC AUC']:+.4f}，Brier {xgb['Brier'] - cox['Brier']:+.4f}。"
        "这些是单一时间切分的点估计，不是显著性结论。", "",
        "## 校准：到某一天还没买的人有多少", "",
        "![测试集 14、30、60 天校准图](cox_calibration.png)", "",
        "每个时间点只看计划随访覆盖该日的测试用户。按各模型预测的未购买概率近似等频分为至多五组"
        "（相同预测值不拆组），比较组内平均预测和实际未购买比例。"
        "窗口内无提前删失，因此实际比例等于该组同一时间点的 Kaplan–Meier 估计。"
        "误差线是观测比例的 95% Wilson 区间，不包含训练不确定性。"
        "点在线下表示高估未购买比例，也就是低估购买率。", "",
        "| 天数 | 可评估人数 | 已购买 | 实际未购买 | Cox B 预测未购买（均值） | 预测减实际 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for day, n, events, pred in coverage:
        observed = 1 - events / n
        lines.append(f"| {day} | {n} | {events} | {observed:.1%} | {pred:.1%} | {(pred-observed)*100:+.1f} pp |")
    lines += [
        "",
        "长窗口排除了更多晚入组用户，不能把上表跨行读成同一批人的实际生存曲线。"
        "没有用测试结果重新校准模型。", "",
        "## 生存模型多给了什么", "",
        "- 30 天二分类回答「会不会在 30 天内买」，不直接回答「什么时候买」。",
        "- Cox 给出整条时间曲线，同一模型可估算不同触达窗口的购买概率。比如第 14 天结束还没买的人，"
        "在第 15 至 30 天购买的模型概率为 `1 - S(30)/S(14)`。这不是触达本身的因果效果。",
        "- Cox 训练保留右删失用户截至截止日的在险信息；固定窗口基线需要确认窗口内的二分类标签。"
        "能给出多时间点预测不等于概率已足够准，投放前应在新时间段核对校准。", "",
        "复跑：`./.venv/Scripts/python.exe -B src/evaluate_cox.py`。分箱明细见 `cox_calibration_bins.csv`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    data, features = tc.load()
    if problems := tc.validate(data, features):
        parser.error("; ".join(problems))
    train = data.loc[data["split"].eq("train")]
    test = data.loc[data["split"].eq("test")]
    c_full, metrics, bins, coverage, train30 = evaluate(train, test, features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(bins, coverage, args.out_dir / "cox_calibration.png")
    bins.to_csv(args.out_dir / "cox_calibration_bins.csv", index=False)
    (args.out_dir / "cox_evaluation.md").write_text(report(test, c_full, metrics, coverage, train30, features),
                                                    encoding="utf-8")
    print(f"Full test Harrell C-index: {c_full}\nShared 30-day cohort: {metrics}")


if __name__ == "__main__":
    main()
