"""构建 Cox 建模表：目标 = 打分后多久发生下一笔有效购买。

约定
----
- 打分日 score_day = entry_date + 14 天，特征只用 [entry_date, score_day) 的交易
- 目标 = 从 score_day 起第一笔有效购买的时间
- E=1 表示观察到下一笔购买；到该 split 的标签截止日仍未购买 -> 右删失（E=0）
- 已在前 14 天复购的客户同样入组，预测的是「下一单」
- 标签截止：train 截止到 t1，val 截止到 t2，test 截止到观察终点；不跨期
- 更早的 14/30 沉默口径留作对照，不在本脚本内

用法  : ./.venv/Scripts/python.exe src/build_cox_features.py
"""
from __future__ import annotations

import bisect
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

OBS_END = pd.Timestamp("2011-12-09")
SCORE_OFFSET_DAYS = 14
GAP_DAYS = 30
SURV = os.path.join("data", "processed", "survival.csv")
RAW = os.path.join("data", "raw", "online_retail_II.xlsx")
OUT_TABLE = os.path.join("data", "processed", "cox_model_table.csv")
REPORTS = "reports"
FEATURE_SPEC = os.path.join(REPORTS, "feature_spec.md")
SELECTION_LOG = os.path.join(REPORTS, "feature_selection_log.md")
LABEL_HEALTH = os.path.join(REPORTS, "label_health.md")

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20


def load_entries():
    df = pd.read_csv(SURV, parse_dates=["entry_date"])
    return {int(r.user_id): r.entry_date.date() for r in df.itertuples()}


def scan_transactions(entries, scores):
    import openpyxl

    buy_dates = defaultdict(set)
    win_inv = defaultdict(dict)
    first_buy = {}
    n_rows = 0

    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    try:
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or len(row) < 7:
                    continue
                invoice, stock, qty, dt, price, cust = row[0], row[1], row[3], row[4], row[5], row[6]
                if cust is None or qty is None or qty <= 0 or price is None or price <= 0 or dt is None:
                    continue
                uid = int(cust)
                if uid not in entries:
                    continue
                d = dt.date() if hasattr(dt, "date") else dt
                n_rows += 1
                buy_dates[uid].add(d)
                if uid not in first_buy or d < first_buy[uid]:
                    first_buy[uid] = d
                if not (entries[uid] <= d < scores[uid]):
                    continue
                rec = win_inv[uid].get(invoice)
                if rec is None:
                    rec = {"time": dt, "date": d, "amount": 0.0, "codes": defaultdict(float)}
                    win_inv[uid][invoice] = rec
                else:
                    if dt < rec["time"]:
                        rec["time"] = dt
                        rec["date"] = d
                rec["amount"] += float(qty) * float(price)
                rec["codes"][str(stock)] += float(qty) * float(price)
    finally:
        wb.close()
    print("SCAN rows=" + str(n_rows) + " users=" + str(len(buy_dates)))
    return buy_dates, win_inv, first_buy


def compute_next_purchase(dates, score_day, deadline):
    """从 score_day 起第一笔有效购买。

    返回 (T, E)。打分日当天下单记 T=1，T 恒为正整数。
    """
    # 打分发生在打分日开始时刻，当天发生的订单记第 1 天，因此 T 为严格正整数
    lo = bisect.bisect_left(dates, score_day)
    if lo < len(dates) and dates[lo] <= deadline:
        return (dates[lo] - score_day).days + 1, 1
    return (deadline - score_day).days + 1, 0


def build_features(entries, scores, buy_dates, win_inv, first_buy):
    rows = []
    skipped_not_scored = 0
    entry_mismatch = 0

    for uid, entry in entries.items():
        score = scores[uid]
        if score > OBS_END.date():
            skipped_not_scored += 1
            continue
        if first_buy.get(uid) != entry:
            entry_mismatch += 1

        invs = win_inv[uid]
        window_amount = sum(r["amount"] for r in invs.values())
        last_buy = max(r["date"] for r in invs.values())
        prod_amount = defaultdict(float)
        for r in invs.values():
            for code, amt in r["codes"].items():
                prod_amount[code] += amt
        top1_share = (max(prod_amount.values()) / window_amount) if window_amount > 0 else np.nan
        # 与第一阶段一致：首购日多张发票时按 (时间, 发票号) 稳定排序取最早
        day_invs = [(r["time"], str(inv), r["amount"])
                    for inv, r in invs.items() if r["date"] == entry]
        first_order_amount = min(day_invs, key=lambda x: (x[0], x[1]))[2] if day_invs else np.nan

        rows.append({
            "user_id": uid,
            "entry_date": pd.Timestamp(entry),
            "score_day": pd.Timestamp(score),
            "n_invoices": len(invs),
            "n_active_days": len({r["date"] for r in invs.values()}),
            "first_order_amount": round(first_order_amount, 2) if first_order_amount == first_order_amount else np.nan,
            "window_amount": round(window_amount, 2),
            "days_since_last_buy": (score - last_buy).days,
            "n_products": len(prod_amount),
            "top1_product_share": round(top1_share, 4),
            "repeated_in_window": int(len(invs) > 1),
        })

    print("SKIP not_scored=" + str(skipped_not_scored) + " entry_mismatch=" + str(entry_mismatch))
    return pd.DataFrame(rows)


def split_by_score_day(df):
    df = df.sort_values("score_day").reset_index(drop=True)
    days = sorted(df["score_day"].unique())
    n_days = len(days)
    i1 = max(1, int(n_days * TRAIN_FRAC))
    i2 = max(i1 + 1, int(n_days * (TRAIN_FRAC + VAL_FRAC)))
    t1 = days[i1 - 1]
    t2 = days[i2 - 1]
    df["split"] = "test"
    df.loc[df["score_day"] <= t1, "split"] = "train"
    df.loc[(df["score_day"] > t1) & (df["score_day"] <= t2), "split"] = "val"
    print("SPLIT_CUTS train<=" + str(pd.Timestamp(t1).date()) + " val<=" + str(pd.Timestamp(t2).date()))
    return df


def vif_values(frame):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = sm.add_constant(frame.values.astype(float), has_constant="add")
    return [float(variance_inflation_factor(X, i + 1)) for i in range(frame.shape[1])]


def select_features(train, cols):
    log = []
    keep = list(cols)

    drops = [c for c in keep if train[c].nunique(dropna=False) <= 1]
    for c in drops:
        log.append({"step": "constant", "dropped": c, "reason": "only one unique value"})
    keep = [c for c in keep if c not in drops]

    drops = [c for c in keep if float(train[c].std(ddof=0)) < 1e-8]
    for c in drops:
        log.append({"step": "near_zero_variance", "dropped": c, "reason": "std < 1e-8"})
    keep = [c for c in keep if c not in drops]

    drops = []
    for i in range(len(keep)):
        if keep[i] in drops:
            continue
        for j in range(i + 1, len(keep)):
            if keep[j] in drops:
                continue
            if np.allclose(train[keep[i]].to_numpy(float), train[keep[j]].to_numpy(float)):
                drops.append(keep[j])
                log.append({"step": "duplicate", "dropped": keep[j], "reason": "identical to " + keep[i]})
    keep = [c for c in keep if c not in drops]

    corr = train[keep].corr().abs()
    dropped = set()
    for i, a in enumerate(keep):
        if a in dropped:
            continue
        for b in keep[i + 1:]:
            if b in dropped:
                continue
            r = float(corr.loc[a, b])
            if r > 0.90:
                dropped.add(b)
                log.append({"step": "high_correlation", "dropped": b,
                            "reason": "|r|=" + format(r, ".3f") + " with " + a})
    keep = [c for c in keep if c not in dropped]

    vif_map = dict(zip(keep, vif_values(train[keep]))) if len(keep) > 1 else {keep[0]: 1.0}
    while len(keep) > 1:
        worst = max(vif_map, key=vif_map.get)
        if vif_map[worst] <= 5.0:
            break
        log.append({"step": "vif", "dropped": worst,
                    "reason": "VIF=" + format(vif_map[worst], ".2f") + " > 5"})
        keep.remove(worst)
        vif_map.pop(worst)
        vif_map = dict(zip(keep, vif_values(train[keep]))) if len(keep) > 1 else {keep[0]: 1.0}

    return keep, log, vif_map


def label_health(df, deadlines):
    lines = ["# 标签体检：打分后下一笔购买", "",
             "标签 = 从打分日起第一笔有效购买的时间；到该 split 的标签截止日仍未购买记右删失。",
             "**删失不等于「永不复购」**，只表示在截止日前没观察到。", "",
             "| split | 标签截止日 | 客户数 | 30 天内已购 | 30 天内未购(观察满) | 30 天结局未知 | 60 天内已购 | 60 天内未购(观察满) | 60 天结局未知 |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        dl = deadlines[sp]
        row = [sp, str(pd.Timestamp(dl).date()), str(len(sub))]
        for h in (30, 60):
            # 与 T 同一计日规则：T>=h 表示「至少观察/等待了 h 天」
            bought = (sub["E"] == 1) & (sub["T"] <= h)
            enough = sub["T"] >= h
            not_bought = enough & ~bought
            unknown = ~enough & ~bought
            total = int(bought.sum()) + int(not_bought.sum()) + int(unknown.sum())
            assert total == len(sub), f"{sp} h={h} 三类不互斥: {total} != {len(sub)}"
            row += [str(int(bought.sum())), str(int(not_bought.sum())), str(int(unknown.sum()))]
        lines.append("| " + " | ".join(row) + " |")
        summary[sp] = {"n": len(sub), "deadline": dl}
    return lines, summary


def main():
    entries = load_entries()
    scores = {u: e + timedelta(days=SCORE_OFFSET_DAYS) for u, e in entries.items()}
    print("USERS " + str(len(entries)))

    buy_dates, win_inv, first_buy = scan_transactions(entries, scores)
    df = build_features(entries, scores, buy_dates, win_inv, first_buy)
    # 防未来数据混入：窗口只有 14 天，任何超出都说明过滤逻辑出错
    assert (df["score_day"] - df["entry_date"]).dt.days.eq(SCORE_OFFSET_DAYS).all()
    assert (df["n_active_days"] <= SCORE_OFFSET_DAYS).all(), "窗口内活跃天数不应超过 14"
    assert (df["days_since_last_buy"] <= SCORE_OFFSET_DAYS).all(), "距最后购买天数应落在窗口内"
    assert (df["n_active_days"] <= df["n_invoices"]).all(), "活跃天数不应多于发票数"
    print("ROWS " + str(len(df)) + " (before labels)")

    df = split_by_score_day(df)

    t1 = df.loc[df["split"] == "train", "score_day"].max()
    t2 = df.loc[df["split"] == "val", "score_day"].max()
    deadlines = {"train": t1.date(), "val": t2.date(), "test": OBS_END.date()}

    ts, es = [], []
    for r in df.itertuples():
        T, E = compute_next_purchase(sorted(buy_dates[int(r.user_id)]),
                                     r.score_day.date(), deadlines[r.split])
        ts.append(T)
        es.append(E)
    df["T"] = ts
    df["E"] = es
    df["label_deadline"] = df["split"].map(
        {k: pd.Timestamp(v) for k, v in deadlines.items()})

    assert int((df["T"] <= 0).sum()) == 0, "T 必须为正整数（打分日当天记第 1 天）"
    print("overall events=" + str(int(df["E"].sum())) + " censored=" + str(int((df["E"] == 0).sum())))

    feature_cols = ["n_invoices", "n_active_days", "first_order_amount", "window_amount",
                    "days_since_last_buy", "n_products", "top1_product_share"]

    train = df[df["split"] == "train"]
    keep, log, final_vif = select_features(train, feature_cols)
    print("FEATURES kept=" + str(keep))

    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    whitelist = {
        "features": keep,
        "dropped": [c for c in feature_cols if c not in keep],
        "target": {"T": "days since score_day, positive integer", "E": "1 observed next purchase, 0 right-censored"},
        "key_columns": ["user_id", "entry_date", "score_day", "label_deadline", "split"],
    }
    whl_path = os.path.join(REPORTS, "feature_whitelist.json")
    with open(whl_path, "w", encoding="utf-8") as f:
        json.dump(whitelist, f, ensure_ascii=False, indent=2)
    print("SAVED " + whl_path + " features=" + str(len(keep)))

    out_cols = ["user_id", "entry_date", "score_day", "label_deadline", "T", "E", "split"] + keep
    df[out_cols].to_csv(OUT_TABLE, index=False)
    print("SAVED " + OUT_TABLE + " rows=" + str(len(df)))

    recon = df.groupby("split").agg(
        n=("user_id", "size"), events=("E", "sum"),
        censored=("E", lambda s: int((s == 0).sum())),
        med_T=("T", "median"),
        score_min=("score_day", "min"), score_max=("score_day", "max"))
    print()
    print(recon.to_string())

    lines, _ = label_health(df, deadlines)
    with open(LABEL_HEALTH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SAVED " + LABEL_HEALTH)

    spec = [
        "# Cox 建模表 · 特征清单",
        "",
        "## 约定",
        "",
        "- **打分日** score_day = entry_date + " + str(SCORE_OFFSET_DAYS) + " 天",
        "- **特征窗口** [entry_date, score_day)",
        "- **计日** 打分发生在打分日开始时刻；当天成交记第 1 天，T 为严格正整数",
        "- **目标** 从 score_day 起第一笔有效购买的时间",
        "- **E=1** 观察到下一笔购买；**E=0** 到该 split 标签截止日仍未购买（右删失，不等于永不复购）",
        "- 已在前 14 天复购的客户同样入组，预测的是「下一单」",
        "- 标签截止：train 到 t1，val 到 t2，test 到观察终点；不跨期",
        "",
        "## 特征表",
        "",
        "| 列 | 来源 | 公式 | 单位 | 窗口 | 缺失 | 打分时可得 |",
        "|---|---|---|---|---|---|---|",
        "| n_invoices | 发票 | 窗口内 distinct Invoice 数 | 张 | 窗口内 | 无 | 是 |",
        "| n_active_days | 交易日期 | 窗口内 distinct 购买日数 | 天 | 窗口内 | 无 | 是 |",
        "| first_order_amount | 首购当日 | 首购日最早一张发票的金额 | GBP | 首购日 | 无 | 是 |",
        "| window_amount | 交易明细 | 窗口内金额合计 | GBP | 窗口内 | 无 | 是 |",
        "| days_since_last_buy | 交易日期 | score_day 减窗口内最后购买日 | 天 | 窗口内 | 无 | 是 |",
        "| n_products | 商品编码 | 窗口内 distinct StockCode 数 | 个 | 窗口内 | 无 | 是 |",
        "| top1_product_share | 商品编码 | 最大商品金额 / window_amount | 比例 | 窗口内 | 分母 0 记 NA | 是 |",
        "",
        "> **品类说明**：原始数据无可靠品类字段，n_products 与 top1_product_share 只是商品广度与集中度的代理，不是品类。",
        "",
        "## 样本对账",
        "",
        "| split | 客户数 | 事件数 | 删失数 | T 中位 | score_day 范围 |",
        "|---|---|---|---|---|---|",
    ]
    for sp in ("train", "val", "test"):
        r = recon.loc[sp]
        spec.append("| " + sp + " | " + format(int(r["n"]), ",") + " | " + format(int(r["events"]), ",") +
                    " | " + format(int(r["censored"]), ",") + " | " + format(float(r["med_T"]), ".0f") +
                    " | " + str(r["score_min"].date()) + " ~ " + str(r["score_max"].date()) + " |")
    spec += ["", "合计 " + format(len(df), ",") + " 人；事件 " + format(int(df["E"].sum()), ",") +
             "，删失 " + format(int((df["E"] == 0).sum()), ",") + "。", "",
             "标签体检见 label_health.md；14/30 沉默口径的旧曲线见第一阶段 reports/km_*.md。"]
    with open(FEATURE_SPEC, "w", encoding="utf-8") as f:
        f.write("\n".join(spec))
    print("SAVED " + FEATURE_SPEC)

    lines2 = ["# 特征筛选记录", "",
              "只在 train 上拟合筛选规则，原样应用到 val / test。", "",
              "## 初始候选（" + str(len(feature_cols)) + " 列）", "",
              "| 列 | train 均值 | train 标准差 | 缺失 |", "|---|---|---|---|"]
    for c in feature_cols:
        lines2.append("| " + c + " | " + format(train[c].mean(), ".4f") + " | " +
                      format(train[c].std(), ".4f") + " | " + str(int(train[c].isna().sum())) + " |")
    lines2 += ["", "## 逐步删除", ""]
    if log:
        lines2 += ["| 步骤 | 删除列 | 原因 |", "|---|---|---|"]
        for item in log:
            lines2.append("| " + item["step"] + " | " + item["dropped"] + " | " + item["reason"] + " |")
    else:
        lines2.append("无删除。")
    lines2 += ["", "## 保留特征与 VIF", "", "| 列 | VIF |", "|---|---|"]
    for c in keep:
        lines2.append("| " + c + " | " + format(final_vif[c], ".3f") + " |")
    corr = train[keep].corr()
    lines2 += ["", "## 保留特征相关矩阵", "", "| | " + " | ".join(keep) + " |",
               "|---|" + "---|" * len(keep)]
    for a in keep:
        lines2.append("| " + a + " | " + " | ".join(format(corr.loc[a, b], ".2f") for b in keep) + " |")
    with open(SELECTION_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))
    print("SAVED " + SELECTION_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
