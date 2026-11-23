"""左截断体检：数据起点之后多久，「首次出现」才可信地代表新客？

数据从 OBS_START 才开始，更早的购买看不到。为了量化「首次出现 ≠ 新客」有多严重，
把第一年藏起来，假装数据从 PSEUDO_START 才开始：在 PSEUDO_START 之后首次出现的用户里，
真正在藏起来的那一年里买过的，就是会被误当成新客的老客户。按「距伪起点的月数」统计该比例，
再看不同回看期下的剩余混入比例。

注意：伪起点之前只有约 12 个月历史，沉睡超过 12 个月的回流客户在这里也会被算成新客，
所以结果是混入比例的下界。

输入  : data/raw/online_retail_II.xlsx
输出  : reports/left_truncation.csv, reports/left_truncation.md

用法  : ./.venv/Scripts/python.exe -B src/check_left_truncation.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

if __package__:
    from .build_survival import COHORT_START, LOOKBACK_DAYS, OBS_START
else:
    from build_survival import COHORT_START, LOOKBACK_DAYS, OBS_START

RAW = os.path.join("data", "raw", "online_retail_II.xlsx")
REPORTS = "reports"
PSEUDO_START = pd.Timestamp("2010-12-01")
LOOKBACKS = (30, 90, 180, 270)  # 伪起点后只有约 12 个月，更长的回看期样本太少


def load_user_days(path: str = RAW) -> pd.DataFrame:
    import openpyxl

    days = set()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in wb.sheetnames:
            for row in wb[sheet].iter_rows(min_row=2, values_only=True):
                if not row or len(row) < 7:
                    continue
                qty, dt, price, cust = row[3], row[4], row[5], row[6]
                if cust is None or qty is None or qty <= 0 or price is None or price <= 0 or dt is None:
                    continue
                days.add((int(cust), dt.date() if hasattr(dt, "date") else dt))
    finally:
        wb.close()
    return pd.DataFrame(sorted(days), columns=["user_id", "date"]).astype({"date": "datetime64[ns]"})


def contamination(days: pd.DataFrame, pseudo_start: pd.Timestamp) -> pd.DataFrame:
    """伪起点后首次出现的用户，按距伪起点的天数，标出其是否在伪起点前买过。"""
    seen_before = set(days.loc[days["date"] < pseudo_start, "user_id"])
    first = days.loc[days["date"] >= pseudo_start].groupby("user_id")["date"].min()
    return pd.DataFrame({
        "days_since_start": (first - pseudo_start).dt.days,
        "returning": first.index.isin(seen_before),
    })


def by_month(table: pd.DataFrame) -> pd.DataFrame:
    month = (table["days_since_start"] // 30).rename("month")
    return table.groupby(month).agg(apparent_new=("returning", "size"),
                                    returning_share=("returning", "mean")).reset_index()


def by_lookback(table: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        sub = table.loc[table["days_since_start"] >= lb]
        rows.append({"lookback_days": lb, "apparent_new": len(sub),
                     "returning_share": sub["returning"].mean() if len(sub) else float("nan")})
    return pd.DataFrame(rows)


def report(months: pd.DataFrame, lookbacks: pd.DataFrame, entry_months: pd.Series) -> str:
    lines = [
        "# 左截断体检：首次出现不等于新客", "",
        f"数据从 {OBS_START.date()} 开始，更早的购买看不到。早期「首次出现」的用户里有大量老客户。",
        f"量化方法：把第一年藏起来，假设数据从 {PSEUDO_START.date()} 才开始，"
        "看伪起点后首次出现的用户中有多少其实在藏起来的那一年里买过。"
        "伪起点前只有约 12 个月历史，所以下表是混入比例的**下界**。", "",
        "## 距数据起点的月数 vs 老客户混入比例", "",
        "| 距伪起点月数 | 表观新客 | 其中老客户 |", "|---:|---:|---:|",
    ]
    lines += [f"| {int(r.month)} | {r.apparent_new} | {r.returning_share:.1%} |"
              for r in months.itertuples(index=False)]
    lines += ["", "## 不同回看期下剩余的老客户比例", "",
              "回看期 L：只保留首次出现在起点 + L 天之后的用户（入组前至少 L 天无购买）。", "",
              "| 回看期 | 保留的表观新客 | 其中曾购买（沉睡 ≥ L 天后回流） |", "|---:|---:|---:|"]
    lines += [f"| {r.lookback_days} 天 | {r.apparent_new} | {r.returning_share:.1%} |"
              for r in lookbacks.itertuples(index=False)]
    lines += ["", "## 真实数据：各月首次出现人数", "",
              "| 月份 | 首次出现人数 |", "|---|---:|"]
    lines += [f"| {m} | {n} |" for m, n in entry_months.items()]
    lines += [
        "", "## 决策", "",
        f"- 采用 **{LOOKBACK_DAYS} 天回看期**：只纳入首次出现不早于 {COHORT_START.date()} 的用户。",
        "- 入组人群因此定义为「入组前至少 "
        f"{LOOKBACK_DAYS} 天无购买的客户」，即**新客或沉睡后回流的客户**，不再声称是纯新客。"
        "上表显示其中仍有相当比例是回流客户；数据只有两年，无法把二者分开。",
        "- 取舍：回看期越长越干净，但样本越少。365 天会让样本降到约一半，"
        "测试集只剩数百人，指标区间过宽；180 天先排除混入最严重的前半年。",
        f"- 修正前数据首月（{entry_months.index[0]}）首次出现 {entry_months.iloc[0]} 人，"
        f"约为随后三个月均值的 {entry_months.iloc[0] / entry_months.iloc[1:4].mean():.1f} 倍；"
        f"按上表，起点后第一个月的表观新客中 {months.iloc[0].returning_share:.0%} 是老客户。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not os.path.exists(RAW):
        print(f"ERROR: {RAW} not found.", file=sys.stderr)
        return 1
    days = load_user_days()
    table = contamination(days, PSEUDO_START)
    months, lookbacks = by_month(table), by_lookback(table)
    entry_months = days.groupby("user_id")["date"].min().dt.to_period("M").value_counts().sort_index()
    os.makedirs(REPORTS, exist_ok=True)
    out = pd.concat([months.assign(table="by_month"), lookbacks.assign(table="by_lookback")])
    out.to_csv(os.path.join(REPORTS, "left_truncation.csv"), index=False)
    with open(os.path.join(REPORTS, "left_truncation.md"), "w", encoding="utf-8") as f:
        f.write(report(months, lookbacks, entry_months))
    print(lookbacks.to_string(index=False))
    print("SAVED reports/left_truncation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
