"""从原始 XLSX 构建用户时间轴。

输入  : data/raw/online_retail_II.xlsx
输出  : data/processed/user_timeline.csv

字段  : user_id, first_date, last_date, active_days, span_days
清洗  : Customer ID 为空 / Quantity <= 0 / Price <= 0 / 日期为空，一律丢弃

本脚本只负责「读取 -> 清洗 -> 按用户聚合」，不包含任何事件判定逻辑。
事件判定在 src/build_survival.py。

用法  : ./.venv/Scripts/python.exe src/build_user_timeline.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import openpyxl

RAW = os.path.join("data", "raw", "online_retail_II.xlsx")
OUT = os.path.join("data", "processed", "user_timeline.csv")


def main() -> int:
    if not os.path.exists(RAW):
        print(f"ERROR: {RAW} not found. Download the dataset first.", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    user_dates: dict[int, set] = defaultdict(set)
    stats = {"total": 0, "no_cust": 0, "nonpos_qty": 0, "nonpos_price": 0, "no_date": 0, "kept": 0}

    try:
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(min_row=2, values_only=True):
                stats["total"] += 1
                if not row or len(row) < 7:
                    continue
                qty, dt, price, cust = row[3], row[4], row[5], row[6]
                if cust is None:
                    stats["no_cust"] += 1
                    continue
                if qty is None or qty <= 0:
                    stats["nonpos_qty"] += 1
                    continue
                if price is None or price <= 0:
                    stats["nonpos_price"] += 1
                    continue
                if dt is None:
                    stats["no_date"] += 1
                    continue
                d = dt.date() if hasattr(dt, "date") else dt
                user_dates[int(cust)].add(d)
                stats["kept"] += 1
    finally:
        wb.close()

    print("STATS", stats)
    print("N_USERS", len(user_dates))

    # ---- QC：失败则退出，不写文件 ----
    errors = []
    if stats["total"] == 0:
        errors.append("raw file has no data rows")
    if len(user_dates) == 0:
        errors.append("no users survived cleaning")
    if stats["kept"] == 0:
        errors.append("kept rows == 0")
    if errors:
        for e in errors:
            print(f"QC_FAIL {e}", file=sys.stderr)
        return 1

    rows = []
    for uid, ds in user_dates.items():
        ds = sorted(ds)
        rows.append((uid, ds[0].isoformat(), ds[-1].isoformat(), len(ds), (ds[-1] - ds[0]).days))
    rows.sort()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "first_date", "last_date", "active_days", "span_days"])
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"SAVED {OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
