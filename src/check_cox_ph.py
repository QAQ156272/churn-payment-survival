"""Phase 2, step 3: training-only PH diagnostics and a stratified sensitivity fit."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

if __package__:
    from . import train_cox as tc
else:
    import train_cox as tc

ALPHA = 0.05
HORIZON = 90
WINDOW_END = 14
STRATUM = "recency_stratum"
TRANSFORMS = ("rank", "km", "log", "identity")


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid PH p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def event_midranks(durations, events, _weights):
    # lifelines stores stratified observations in stratum order, not time order.
    event_days = np.sort(durations[events].to_numpy())
    left = event_days.searchsorted(durations, side="left")
    right = event_days.searchsorted(durations, side="right")
    return pd.Series((left + right + 1) / 2, index=durations.index)


def diagnose(model, frame, name):
    event_index = model.event_observed[model.event_observed].index
    residuals = model.compute_residuals(frame, kind="scaled_schoenfeld").reindex(event_index)
    if not np.isfinite(residuals).all().all():
        raise ValueError("Invalid or unaligned Schoenfeld residuals")
    tables = []
    for transform in TRANSFORMS:
        time_transform = event_midranks if transform == "rank" else transform
        test = proportional_hazard_test(
            model, frame, time_transform=time_transform, precomputed_residuals=residuals,
        )
        result = test.summary[["test_statistic", "p"]].rename_axis("feature").reset_index()
        result["p_holm"] = holm(result["p"])
        result["reject_05"] = result["p_holm"] < ALPHA
        result["transform"] = transform
        result["model"] = name
        tables.append(result)
    return pd.concat(tables, ignore_index=True)[
        ["model", "transform", "feature", "test_statistic", "p", "p_holm", "reject_05"]
    ]


def with_stratum(frame):
    recency = frame[tc.BASELINE]
    if not recency.between(1, WINDOW_END).all():
        raise ValueError("Recency lies outside the 14-day scoring window")
    return frame.assign(**{STRATUM: np.where(recency == WINDOW_END, "14", "1-13")})


def survival_score(model, predictors):
    # S(90) includes the applicable baseline hazard, unlike partial hazard.
    scores = model.predict_survival_function(predictors, times=[HORIZON]).iloc[0]
    scores = scores.reindex(predictors.index)
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Invalid survival predictions")
    return scores


def run(train, val, features):
    columns = ["T", "E", *features]
    train = train[columns].reset_index(drop=True)
    val = val[columns].reset_index(drop=True)
    baseline, issues = tc.fit(train, features, penalizer=0)
    if issues:
        raise ValueError("Baseline fit warnings: " + "; ".join(issues))

    other = [feature for feature in features if feature != tc.BASELINE]
    strat_columns = [*other, STRATUM]
    strat_train = with_stratum(train)[["T", "E", *strat_columns]]
    strat_val = with_stratum(val)
    support = strat_train.assign(at_90=strat_train["T"].ge(HORIZON)).groupby(STRATUM).agg(
        train_n=("E", "size"), train_events=("E", "sum"), at_90=("at_90", "sum"),
    )
    if len(support) != 2 or (support[["train_events", "at_90"]] <= 0).any().any():
        raise ValueError("Insufficient training support for both recency strata")
    support["val_n"] = strat_val.groupby(STRATUM).size().reindex(support.index, fill_value=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        stratified = CoxPHFitter(strata=[STRATUM], penalizer=0, l1_ratio=0)
        stratified.fit(strat_train, duration_col="T", event_col="E")

    diagnostics = pd.concat([
        diagnose(baseline, train, "B"),
        diagnose(stratified, strat_train, "B_stratified"),
    ], ignore_index=True)
    original_c = baseline.score(val, scoring_method="concordance_index")
    c_baseline = concordance_index(val["T"], survival_score(baseline, val[features]), val["E"])
    c_strat = concordance_index(val["T"], survival_score(stratified, strat_val[strat_columns]), val["E"])
    return diagnostics, support, (original_c, c_baseline, c_strat)


def report(diagnostics, support, scores, train, val, features):
    original_c, c_baseline, c_strat = scores
    lines = [
        "# 第二阶段第 3 步：Cox 比例风险诊断与分层实验",
        "",
        "目标仍为首购后第 14 天打分起，到下一笔有效购买的时间。当天成交记 T=1。"
        "训练集拟合和诊断，验证集只做排序对照；测试集未参与拟合、诊断、方案选择或评估。",
        "",
        f"- 训练集：{len(train):,} 人，{int(train['E'].sum()):,} 个事件；"
        f"验证集：{len(val):,} 人，{int(val['E'].sum()):,} 个事件。",
        f"- 复现模型 B：原六个特征、无惩罚、不做变换，lifelines {lifelines.__version__}。",
        "- 特征：" + ", ".join(features) + "。",
        "- 主检验：缩放 Schoenfeld 残差，按实际事件日取全局时间秩（rank），同日事件取平均秩。"
        "分层后也不按层的排列顺序累积事件。每个模型内按特征做 Holm 校正，α=0.05。"
        "KM、log 和 identity 变换是敏感性检查，各自单独校正，不用于选择特征或分层方案。"
        "检验使用 lifelines 的近似统计量，未拒绝不等于证明 PH 成立。",
        "",
        "## 主检验结果（训练集）",
        "",
        "| 模型 | 特征 | 统计量 | 原始 p | Holm p | 拒绝 PH |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in diagnostics[diagnostics["transform"].eq("rank")].itertuples(index=False):
        lines.append(f"| {row.model} | {row.feature} | {row.test_statistic:.4f} | "
                     f"{row.p:.4g} | {row.p_holm:.4g} | {'是' if row.reject_05 else '否'} |")
    lines += [
        "",
        "## 分层敏感性实验",
        "",
        "按打分时的购后间隔分为 1–13 天和 14 天。14 天是特征观察窗边界，"
        "表示首购日以后、打分以前没有再购买；没有按验证集表现搜索切点。"
        "两层各拟合一条基准风险，其他五个系数在两层共用；分层后的购后间隔没有单一 HR。"
        "这个对照也移除了层内间隔的线性效应，因此不能把性能变化单独归因于 PH。",
        "",
        "| 层 | 训练人数 | 训练事件 | 训练 T≥90 人数 | 验证人数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in support.reset_index().itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    lines += [
        "",
        "## 验证集排序（Harrell C-index）",
        "",
        "两个模型使用同一批人、同一原始随访 T/E 和相同的可比较对定义。"
        "主对照统一按预测 S(90) 排序，越低表示预计到第 90 天结束仍未购买的概率越低。"
        "这不是只评 90 天内的 C-index，也不是 90 天购买概率的校准检验。",
        "",
        "| 模型 | 排序分数 | 验证 C-index |",
        "|---|---|---:|",
        f"| 原 B（第 2 步口径） | 部分风险 | {original_c:.6f} |",
        f"| 原 B（同口径对照） | S(90) | {c_baseline:.6f} |",
        f"| 分层 B | 层特异 S(90) | {c_strat:.6f} |",
        "",
        f"分层 B − 原 B（同口径）= {c_strat - c_baseline:+.6f}。"
        "这是单个验证集的点估计，不是显著性结论。",
        "",
        "## 结论",
        "",
    ]
    for model in ("B", "B_stratified"):
        rows = diagnostics[diagnostics["model"].eq(model)]
        primary = rows[rows["transform"].eq("rank")]
        sensitivity = rows[~rows["transform"].eq("rank")]
        lines.append(f"- {model}：主检验 {int(primary['reject_05'].sum())}/{len(primary)} 项拒绝；"
                     f"敏感性检验 {int(sensitivity['reject_05'].sum())}/{len(sensitivity)} 项拒绝。")
    if not diagnostics.loc[diagnostics["model"].eq("B"), "reject_05"].any():
        lines.append("- 保留原 B 作为下一步 Cox 候选。没有检出需要修正的 PH 违背，分层仅作敏感性对照。"
                     "原模型 HR 应表述为拟合模型下的瞬时复购率比，不是购买概率或因果效果。")
    else:
        lines.append("- 原 B 有 PH 警报。分层只调整购后间隔，不能替代对其他告警特征的复核。"
                     "是否替换主模型须结合剩余诊断结果，不能只看验证 C-index。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    data, features = tc.load()
    if problems := tc.validate(data, features):
        parser.error("; ".join(problems))
    train = data.loc[data["split"].eq("train")]
    val = data.loc[data["split"].eq("val")]
    diagnostics, support, scores = run(train, val, features)
    notes = report(diagnostics, support, scores, train, val, features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.out_dir / "cox_ph_results.csv", index=False)
    (args.out_dir / "cox_ph_experiment.md").write_text(notes, encoding="utf-8")
    print(f"Validation C-index: original={scores[0]:.6f}, baseline_S90={scores[1]:.6f}, "
          f"stratified_S90={scores[2]:.6f}")


if __name__ == "__main__":
    main()
