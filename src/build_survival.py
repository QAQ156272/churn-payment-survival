"""构造生存分析表 (T, E)。

输入  : data/processed/user_timeline.csv
输出  : data/processed/survival.csv

口径  : 流失 = 末次购买后连续 N 天无购买（主口径 N=30，见 spec.md）
删失  : 到观察终点仍未达到 N 天，E=0
时间  : T 采用 inclusive 计日（首尾同日 = 1 天），保证 T >= 1
入组  : 数据从 OBS_START 才开始，更早的购买看不到（左截断）。首次「出现」在
        OBS_START + LOOKBACK_DAYS 之前的用户无法确认此前没有购买，大量是老客户，
        一律剔除。保留下来的人在入组前至少 LOOKBACK_DAYS 天无购买：
        真正的新客，或沉睡 >= LOOKBACK_DAYS 天后回流的客户。
        依据见 src/check_left_truncation.py 与 reports/left_truncation.md。

QC 是硬门禁：任何一项检查失败 -> 返回非零 -> 不写文件，旧输出保留。
输入先按字符串读入，逐字段校验格式，再解析，避免 "20110101" 这类值
被 pandas 当成 Unix 时间戳静默接受。

用法  : ./.venv/Scripts/python.exe src/build_survival.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

OBS_START = pd.Timestamp("2009-12-01")
OBS_END = pd.Timestamp("2011-12-09")
MAIN_N = 30
LOOKBACK_DAYS = 180
COHORT_START = OBS_START + pd.Timedelta(days=LOOKBACK_DAYS)
SENS_N = (7, 14, 30)
IN = os.path.join("data", "processed", "user_timeline.csv")
OUT = os.path.join("data", "processed", "survival.csv")

REQUIRED_COLS = ["user_id", "first_date", "last_date"]
ID_RE = re.compile(r"^\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """按阈值 n_days 生成 (T, E)。"""
    out = df.copy()
    out["gap"] = (OBS_END - out["last_date"]).dt.days
    out["E"] = (out["gap"] >= n_days).astype(int)

    t_event = (out["last_date"] - out["first_date"]).dt.days + 1   # inclusive
    t_cens = (OBS_END - out["first_date"]).dt.days + 1             # inclusive
    out["T"] = np.where(out["E"] == 1, t_event, t_cens)
    return out


def apply_lookback(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """剔除首次出现早于 COHORT_START 的用户（入组前回看期不满）。"""
    keep = df["first_date"] >= COHORT_START
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def validate_structure(raw: pd.DataFrame) -> list[str]:
    """对「按字符串读入」的原始表做格式校验。"""
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(raw) == 0:
        return ["input table is empty"]

    problems: list[str] = []

    uid = raw["user_id"].astype(str)
    bad_id = raw["user_id"].isna() | ~uid.str.match(ID_RE)
    if bad_id.any():
        problems.append(
            f"user_id not integer: {int(bad_id.sum())} e.g. {raw.loc[bad_id, 'user_id'].head(3).tolist()}"
        )

    for col in ("first_date", "last_date"):
        s = raw[col].astype(str)
        bad = raw[col].isna() | ~s.str.match(DATE_RE)
        if bad.any():
            problems.append(
                f"{col} not YYYY-MM-DD: {int(bad.sum())} e.g. {raw.loc[bad, col].head(3).tolist()}"
            )

    return problems


def validate_dates(df: pd.DataFrame) -> list[str]:
    """日期解析之后的内容校验。"""
    if len(df) == 0:
        return ["input table is empty"]

    problems: list[str] = []
    dup = int(df["user_id"].duplicated().sum())
    if dup:
        problems.append(f"duplicated user_id: {dup}")
    for col in ("first_date", "last_date"):
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"{col} unparseable: {n}")
    # 回看期过滤之前先拦越界日期，避免坏数据被过滤「静默吞掉」
    for col in ("first_date", "last_date"):
        n = int(((df[col] < OBS_START) | (df[col] > OBS_END)).sum())
        if n:
            problems.append(f"{col} outside [{OBS_START.date()}, {OBS_END.date()}]: {n}")
    return problems


def validate_output(df: pd.DataFrame) -> list[str]:
    if len(df) == 0:
        return ["output table is empty"]

    problems: list[str] = []
    if int(df["T"].isna().sum()):
        problems.append(f"T NaN: {int(df['T'].isna().sum())}")
    if int((df["T"] <= 0).sum()):
        problems.append(f"T <= 0: {int((df['T'] <= 0).sum())}")
    if int((df["T"] < 0).sum()):
        problems.append(f"T < 0: {int((df['T'] < 0).sum())}")
    if int((~df["E"].isin([0, 1])).sum()):
        problems.append(f"E not binary: {int((~df['E'].isin([0, 1])).sum())}")
    if int((df["last_date"] < df["first_date"]).sum()):
        problems.append(f"last_date < first_date: {int((df['last_date'] < df['first_date']).sum())}")
    if int((df["first_date"] < OBS_START).sum()):
        problems.append(
            f"first_date < OBS_START({OBS_START.date()}): {int((df['first_date'] < OBS_START).sum())}"
        )
    if int((df["first_date"] < COHORT_START).sum()):
        problems.append(
            f"first_date < COHORT_START({COHORT_START.date()}): {int((df['first_date'] < COHORT_START).sum())}"
        )
    if int((df["last_date"] < OBS_START).sum()):
        problems.append(f"last_date < OBS_START({OBS_START.date()}): {int((df['last_date'] < OBS_START).sum())}")
    if int((df["first_date"] > OBS_END).sum()):
        problems.append(f"first_date > OBS_END: {int((df['first_date'] > OBS_END).sum())}")
    if int((df["last_date"] > OBS_END).sum()):
        problems.append(f"last_date > OBS_END: {int((df['last_date'] > OBS_END).sum())}")
    if int(df["user_id"].duplicated().sum()):
        problems.append(f"duplicated user_id in output: {int(df['user_id'].duplicated().sum())}")
    return problems


def _abort(problems: list[str]) -> int:
    for p in problems:
        print(f"QC_FAIL {p}", file=sys.stderr)
    print("ABORT: output untouched.", file=sys.stderr)
    return 1


def main() -> int:
    if not os.path.exists(IN):
        print(f"ERROR: {IN} not found. Run src/build_user_timeline.py first.", file=sys.stderr)
        return 1

    try:
        raw = pd.read_csv(IN, dtype=str)
    except pd.errors.EmptyDataError:
        return _abort(["input file is empty (0 bytes, no header)"])
    except pd.errors.ParserError as exc:
        return _abort([f"malformed CSV: {exc}"])

    problems = validate_structure(raw)
    if problems:
        return _abort([f"input: {p}" for p in problems])

    df = raw.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["first_date"] = pd.to_datetime(df["first_date"], format="%Y-%m-%d", errors="coerce")
    df["last_date"] = pd.to_datetime(df["last_date"], format="%Y-%m-%d", errors="coerce")

    problems = validate_dates(df)
    if problems:
        return _abort([f"input: {p}" for p in problems])

    print(f"N_USERS_INPUT {len(df)}")
    df, n_excluded = apply_lookback(df)
    print(f"LOOKBACK {LOOKBACK_DAYS}d cohort_start={COHORT_START.date()} "
          f"excluded={n_excluded} kept={len(df)}")
    if len(df) == 0:
        return _abort(["no users left after lookback filter"])
    main_df = build(df, MAIN_N)

    problems = validate_output(main_df)
    if problems:
        return _abort([f"output: {p}" for p in problems])

    # ---- 原子写 ----
    out = main_df[["user_id", "first_date", "last_date", "T", "E"]].rename(
        columns={"first_date": "entry_date", "last_date": "last_purchase_date"}
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    print(f"SAVED {OUT} rows={len(out)}")

    print()
    print("=== MAIN (N=30) ===")
    print(f"  events   = {int(main_df['E'].sum())} ({main_df['E'].mean():.4f})")
    print(f"  censored = {int((main_df['E'] == 0).sum())} ({(main_df['E'] == 0).mean():.4f})")
    print(f"  T median = {main_df['T'].median():.1f}  min={main_df['T'].min()} max={main_df['T'].max()}")

    print()
    print("=== SENSITIVITY ===")
    print(f"{'N':>3} {'events':>7} {'event_rate':>11} {'medT_evt':>9} {'medT_cens':>10}")
    for n_days in SENS_N:
        d = build(df, n_days)
        print(f"{n_days:>3} {int(d['E'].sum()):>7} {d['E'].mean():>11.4f} "
              f"{d.loc[d['E'] == 1, 'T'].median():>9.0f} {d.loc[d['E'] == 0, 'T'].median():>10.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
