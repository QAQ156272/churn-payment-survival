"""训练 Cox 模型：目标 = 打分后到下一笔有效购买的时间。

模型 A: 只用 days_since_last_buy
模型 B: 白名单全部特征
只在 train 上拟合；val 上做 C-index 排序对照；不使用 test。

用法  : ./.venv/Scripts/python.exe src/train_cox.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

TABLE = os.path.join("data", "processed", "cox_model_table.csv")
WHITELIST = os.path.join("reports", "feature_whitelist.json")
OUT_CSV = os.path.join("reports", "cox_results.csv")
OUT_MD = os.path.join("reports", "cox_training_notes.md")

# HR 对应的业务增量：(增量, 单位说明)
INCREMENTS = {
    "days_since_last_buy": (1.0, "+1 天"),
    "first_order_amount": (100.0, "+100 GBP"),
    "window_amount": (100.0, "+100 GBP"),
    "n_invoices": (1.0, "+1 张发票"),
    "n_products": (1.0, "+1 个商品"),
    "top1_product_share": (0.10, "+10 个百分点"),
}
DEFAULT_INC = (1.0, "+1 单位")
KEY_COLUMNS = {"user_id", "entry_date", "score_day", "label_deadline", "split"}
TARGET_COLUMNS = {"T", "E"}
BASELINE = "days_since_last_buy"
PENALTIES = (0.01, 0.1)


def load():
    df = pd.read_csv(TABLE, parse_dates=["entry_date", "score_day", "label_deadline"])
    with open(WHITELIST, encoding="utf-8") as f:
        wl = json.load(f)
    return df, list(wl["features"])


def validate(df, feats):
    problems = []
    forbidden = set(feats) & (KEY_COLUMNS | TARGET_COLUMNS)
    if forbidden:
        problems.append(f"label or metadata in feature whitelist: {sorted(forbidden)}")
    if len(feats) != len(set(feats)):
        problems.append("duplicate features in whitelist")
    if BASELINE not in feats:
        problems.append(f"missing baseline feature in whitelist: {BASELINE}")
    missing = (KEY_COLUMNS | TARGET_COLUMNS | set(feats)) - set(df.columns)
    if missing:
        return problems + [f"missing columns: {sorted(missing)}"]
    for c in KEY_COLUMNS:
        if df[c].isna().any():
            problems.append(f"{c} has missing values")
    if int(df["user_id"].duplicated().sum()):
        problems.append("duplicated user_id: " + str(int(df["user_id"].duplicated().sum())))
    t = pd.to_numeric(df["T"], errors="coerce")
    if not np.isfinite(t).all():
        problems.append("T must be finite and numeric")
    if int((t <= 0).sum()):
        problems.append("T <= 0: " + str(int((t <= 0).sum())))
    if int((t != t.round(0)).sum()):
        problems.append("T not integer: " + str(int((t != t.round(0)).sum())))
    if int((~df["E"].isin([0, 1])).sum()):
        problems.append("E not in {0,1}")
    for c in set(feats) - forbidden:
        if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all():
            problems.append(c + " must be finite and numeric")
    splits = ("train", "val", "test")
    if not df["split"].isin(splits).all():
        problems.append("unrecognized split")
    cohorts = {name: df[df["split"] == name] for name in splits}
    for name, part in cohorts.items():
        if part.empty:
            problems.append(f"empty split: {name}")
        elif part["label_deadline"].nunique() != 1:
            problems.append(f"{name} must have one label_deadline")
    if any(not pd.api.types.is_datetime64_any_dtype(df[c]) for c in
           ("entry_date", "score_day", "label_deadline")):
        return problems + ["invalid date column"]
    for earlier, later in zip(splits, splits[1:]):
        if cohorts[earlier]["label_deadline"].max() >= cohorts[later]["score_day"].min():
            problems.append(f"{earlier} label_deadline overlaps {later} score_day")
    max_t = (df["label_deadline"] - df["score_day"]).dt.days + 1
    if int((t > max_t).sum()):
        problems.append("T exceeds label_deadline: " + str(int((t > max_t).sum())))
    if ((df["E"] == 0) & (t != max_t)).any():
        problems.append("censored T must equal inclusive follow-up to label_deadline")
    return problems


def fit(train, feats, penalizer=0.0):
    frame = train[["T", "E"] + feats].copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
        cph.fit(frame, duration_col="T", event_col="E")
    conv = [str(w.message) for w in caught
            if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))]
    return cph, conv


def hr_rows(cph, feats, model):
    rows = []
    for f in feats:
        s = cph.summary.loc[f]
        delta, unit = INCREMENTS.get(f, DEFAULT_INC)
        coef = float(s["coef"])
        lo_log = float(s["coef lower 95%"])
        hi_log = float(s["coef upper 95%"])
        rows.append({
            "model": model,
            "feature": f,
            "increment": unit,
            "delta": delta,
            "coef_per_unit": coef,
            "HR": float(np.exp(coef * delta)),
            "CI_low": float(np.exp(lo_log * delta)),
            "CI_high": float(np.exp(hi_log * delta)),
            "p_value": float(s["p"]),
        })
    return rows


def checked_fit(train, val, feats, penalty):
    models, rows, scores, issues = [], [], [], []
    for name, columns in zip(("A: days_since_last_buy only", "B: all whitelisted features"), feats):
        try:
            model, diagnostics = fit(train, columns, penalty)
            if diagnostics:
                issues.extend(f"{name}: {message}" for message in diagnostics)
                continue
            result = pd.DataFrame(hr_rows(model, columns, name))
            values = result[["coef_per_unit", "HR", "CI_low", "CI_high", "p_value"]]
            valid = (np.isfinite(values).all().all()
                     and (result["CI_low"] > 0).all()
                     and (result["CI_low"] <= result["HR"]).all()
                     and (result["HR"] <= result["CI_high"]).all()
                     and result["p_value"].between(0, 1).all())
            score = float(model.score(val[["T", "E"] + columns], scoring_method="concordance_index"))
            if not valid or not np.isfinite(score):
                issues.append(f"{name}: invalid HR, confidence interval, p-value, or validation score")
                continue
            models.append(model)
            rows.append(result)
            scores.append(score)
        except (ConvergenceError, FloatingPointError, ZeroDivisionError) as exc:
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
    if issues:
        return None, issues
    return (models, pd.concat(rows, ignore_index=True), scores), []


def choose_fit(train, val, feats):
    attempts, candidates = [], []
    for penalty in (0.0, *PENALTIES):
        result, issues = checked_fit(train, val, feats, penalty)
        attempts.append((penalty, issues))
        if result is not None:
            candidates.append((penalty, result))
            if penalty == 0:
                break
    if not candidates:
        return None, attempts
    return max(candidates, key=lambda item: item[1][2][1]), attempts


def main():
    df, feats = load()
    print("WHITELIST " + str(feats))

    problems = validate(df, feats)
    if problems:
        for p in problems:
            print("QC_FAIL " + p, file=sys.stderr)
        print("ABORT: input failed QC, no model published.", file=sys.stderr)
        return 1
    print("QC_OK rows=" + str(len(df)))

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    print("TRAIN n=" + str(len(train)) + " events=" + str(int(train["E"].sum())) +
          " censored=" + str(int((train["E"] == 0).sum())))
    print("VAL   n=" + str(len(val)) + " events=" + str(int(val["E"].sum())) +
          " censored=" + str(int((val["E"] == 0).sum())))

    feats_a = [BASELINE]
    selected, attempts = choose_fit(train, val, (feats_a, feats))
    for penalty, issues in attempts:
        print(f"FIT penalizer={penalty:g}: " + ("; ".join(issues) if issues else "accepted"))
    if selected is None:
        print("ABORT: no valid Cox fit; existing reports untouched.", file=sys.stderr)
        return 1
    penalty, ((cph_a, cph_b), res, (c_a, c_b)) = selected
    penalty_note = ("无惩罚（penalizer=0）" if penalty == 0 else
                    f"L2 penalizer={penalty:g}，l1_ratio=0；HR 区间和 p 值为近似结果")

    c_a_tr = float(cph_a.score(train[["T", "E"] + feats_a], scoring_method="concordance_index"))
    c_b_tr = float(cph_b.score(train[["T", "E"] + feats], scoring_method="concordance_index"))
    print("CINDEX train A=" + format(c_a_tr, ".4f") + " B=" + format(c_b_tr, ".4f"))
    print("CINDEX val   A=" + format(c_a, ".4f") + " B=" + format(c_b, ".4f") +
          " delta=" + format(c_b - c_a, ".4f"))

    import lifelines
    import scipy
    versions = {
        "lifelines": lifelines.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    lines = [
        "# Cox 训练记录：打分后到下一笔购买的时间",
        "",
        "## 数据",
        "",
        "- 表：data/processed/cox_model_table.csv",
        "- 目标：从首购后第 14 天打分日起，到下一笔有效购买的时间",
        "- 打分日当天成交记 T=1；E=1 观察到下一单，E=0 右删失",
        "",
        "| split | 人数 | 事件 | 删失 |",
        "|---|---|---|---|",
        "| train | " + str(len(train)) + " | " + str(int(train["E"].sum())) + " | " + str(int((train["E"] == 0).sum())) + " |",
        "| val | " + str(len(val)) + " | " + str(int(val["E"].sum())) + " | " + str(int((val["E"] == 0).sum())) + " |",
        "",
        "测试集未参与任何拟合或指标计算。",
        "",
        "## 配置",
        "",
        "- 模型 A 特征：days_since_last_buy",
        "- 模型 B 特征（白名单全量，顺序）：" + ", ".join(feats),
        "- 惩罚：" + penalty_note,
        "- 变换：无对数变换、无标准化，HR 直接按业务增量换算",
        "- 依赖：" + ", ".join(k + "=" + v for k, v in versions.items()),
        "",
        "## 收敛",
        "",
    ]
    for tried, issues in attempts:
        outcome = "未采用：" + "; ".join(issues) if issues else "无收敛或数值警告，通过校验"
        lines.append(f"- penalizer={tried:g}：{outcome}")

    lines += [
        "",
        "## 验证集 C-index（Harrell）",
        "",
        "| 模型 | train C-index | val C-index |",
        "|---|---|---|",
        "| A（单变量） | " + format(c_a_tr, ".4f") + " | " + format(c_a, ".4f") + " |",
        "| B（全特征） | " + format(c_b_tr, ".4f") + " | " + format(c_b, ".4f") + " |",
        "",
        "**B − A = " + format(c_b - c_a, ".4f") + "**",
        "",
        "## HR 表",
        "",
        "| 模型 | 特征 | 增量 | HR | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append("| " + r["model"] + " | " + r["feature"] + " | " + r["increment"] +
                     " | " + format(r["HR"], ".3f") + " | [" + format(r["CI_low"], ".3f") +
                     ", " + format(r["CI_high"], ".3f") + "] | " + format(r["p_value"], ".4g") + " |")

    lines += ["", "## 业务解读（基于本次拟合）", ""]
    lines.append(f"1. 本次验证集排序：A={c_a:.4f}，B={c_b:.4f}，差值 {c_b - c_a:+.4f}。"
                 "这是验证集点估计，尚不是测试集结论。")
    lines.append("")
    for number, feature in enumerate((BASELINE, "n_invoices"), start=2):
        matches = res[res["model"].str.startswith("B:") & (res["feature"] == feature)]
        if matches.empty:
            continue
        row = matches.iloc[0]
        if row.CI_low > 1:
            direction = f"下一单的瞬时发生率高约 {(row.HR - 1) * 100:.1f}%"
        elif row.CI_high < 1:
            direction = f"下一单的瞬时发生率低约 {(1 - row.HR) * 100:.1f}%"
        else:
            direction = "区间包含 1，不据此判断方向"
        lines.extend([f"{number}. 全特征模型中，其他变量相同的条件下，{feature} {row.increment}："
                      f"HR={row.HR:.3f}（95% CI [{row.CI_low:.3f}, {row.CI_high:.3f}]），{direction}。", ""])
    lines += [
        "## 读法（务必按此解释）",
        "",
        "- HR > 1：在其他入模变量相同时，该变量每增加一个单位增量，**发生下一单的瞬时率更高**",
        "- HR < 1：瞬时率更低",
        "- **不是**购买概率的变化，**不是**流失风险，**不是**因果效果",
        "- 95% CI 跨 1 时，不认定方向",
        "",
        "## 未做",
        "",
        "- 比例风险假设检验（Schoenfeld 残差）留到第 3 步",
        "- 测试集评估、校准、XGBoost 对比不在本步",
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print("SAVED " + OUT_CSV + " rows=" + str(len(res)))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SAVED " + OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
