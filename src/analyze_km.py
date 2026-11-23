"""Kaplan-Meier 总体与分组曲线 + log-rank 检验。

输入  : data/processed/survival.csv      (user_id, entry_date, last_purchase_date, T, E)
        data/raw/online_retail_II.xlsx    (计算「最早一张有效发票」的金额)
输出  : reports/km_overall.png / km_by_group.png / km_summary.csv / logrank.csv / km_notes.md
        data/processed/first_invoice_amount.csv

设计  : 1) 总体先交付, 不依赖分组
        2) 分组先「计算」后「发布」: 拟合与检验全部成功才写分组产物
        3) 失败时保留总体, 并在 notes 里逐文件标注状态

用法  : ./.venv/Scripts/python.exe src/analyze_km.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

OBS_START = pd.Timestamp("2009-12-01")
OBS_END = pd.Timestamp("2011-12-09")
N_DAYS = 30
SURV = os.path.join("data", "processed", "survival.csv")
RAW = os.path.join("data", "raw", "online_retail_II.xlsx")
INVOICE_AMOUNT = os.path.join("data", "processed", "first_invoice_amount.csv")
OLD_AMOUNT = os.path.join("data", "processed", "first_order_amount.csv")
REPORTS = "reports"
CHECKPOINTS = (30, 60)
BINS = (200.0, 500.0)
GROUP_LABELS = ["Low (<=200)", "Mid (200-500)", "High (>500)"]
COLORS = ["#2E86AB", "#A23B72", "#F18F01"]
BT = chr(96)

GROUP_ARTIFACTS = ["km_by_group.png", "group_migration.csv"]


# ---------------------------------------------------------------- 校验
def validate(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if len(df) == 0:
        return ["empty input"]
    if int(df["user_id"].duplicated().sum()):
        problems.append(f"duplicated user_id: {int(df['user_id'].duplicated().sum())}")
    for c in ("entry_date", "last_purchase_date"):
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{c} null: {n}")
    if int((df["entry_date"] > df["last_purchase_date"]).sum()):
        problems.append("entry_date > last_purchase_date")

    t = df["T"]
    if int((~(t > 0)).sum()):
        problems.append(f"T <= 0: {int((~(t > 0)).sum())}")
    if int((t != t.round(0)).sum()):
        problems.append(f"T not integer: {int((t != t.round(0)).sum())}")
    if int((~df["E"].isin([0, 1])).sum()):
        problems.append("E not in {0,1}")

    gap = (OBS_END - df["last_purchase_date"]).dt.days
    e_exp = (gap >= N_DAYS).astype(int)
    t_exp = np.where(
        e_exp == 1,
        (df["last_purchase_date"] - df["entry_date"]).dt.days + 1,
        (OBS_END - df["entry_date"]).dt.days + 1,
    )
    if int((e_exp != df["E"]).sum()):
        problems.append(f"E mismatch vs N={N_DAYS} formula: {int((e_exp != df['E']).sum())}")
    if int((t_exp != df["T"]).sum()):
        problems.append(f"T mismatch: {int((t_exp != df['T']).sum())}")
    return problems


# ---------------------------------------------------------------- 最早发票
def extract_first_invoice(rows, users: set[int]) -> pd.DataFrame:
    """从交易明细里抽出每个用户「最早一张有效发票」的金额。"""
    inv_map: dict[int, dict] = defaultdict(dict)
    for invoice, qty, dt, price, cust in rows:
        if cust is None or qty is None or qty <= 0 or price is None or price <= 0 or dt is None:
            continue
        uid = int(cust)
        if uid not in users:
            continue
        rec = inv_map[uid].get(invoice)
        if rec is None:
            inv_map[uid][invoice] = [dt, float(qty) * float(price)]
        else:
            if dt < rec[0]:
                rec[0] = dt
            rec[1] += float(qty) * float(price)

    out = []
    for uid, invs in inv_map.items():
        min_t = min(v[0] for v in invs.values())
        tied = sorted([k for k, v in invs.items() if v[0] == min_t], key=lambda k: str(k))
        chosen = tied[0]
        out.append({
            "user_id": uid,
            "first_invoice_time": min_t,
            "invoice": str(chosen),
            "first_invoice_amount": round(invs[chosen][1], 2),
            "tied_time_invoices": len(tied),
        })
    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values("user_id").reset_index(drop=True)
    return df


def load_first_invoice(users: set[int]) -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)

    def rows():
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or len(row) < 7:
                    continue
                yield row[0], row[3], row[4], row[5], row[6]

    try:
        return extract_first_invoice(rows(), users)
    finally:
        wb.close()


def assign_group(amount: float) -> int:
    if amount <= BINS[0]:
        return 0
    if amount <= BINS[1]:
        return 1
    return 2


# ---------------------------------------------------------------- KM 工具
def step_value(kmf: KaplanMeierFitter, t: float) -> tuple[float, float, float]:
    """KM 阶梯在 t 处的取值，不做线性插值；超出随访范围返回 NaN。"""
    sf = kmf.survival_function_
    idx = sf.index.values
    if len(idx) == 0 or t < idx[0] or t > idx[-1]:
        return np.nan, np.nan, np.nan
    pos = int(np.searchsorted(idx, t, side="right") - 1)
    if pos < 0:
        return np.nan, np.nan, np.nan
    s = float(sf.iloc[pos, 0])
    ci = kmf.confidence_interval_
    return s, float(ci.iloc[pos, 0]), float(ci.iloc[pos, 1])


def at_risk(t: np.ndarray, times: list[int]) -> list[int]:
    return [int((t >= x).sum()) for x in times]


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        val = max(val, prev)
        adj[i] = val
        prev = val
    return adj


def decide_test_mode(groups_present: list[int]) -> str:
    n = len(groups_present)
    if n < 2:
        return "not_tested"
    if n == 2:
        return "two_group"
    return "multi_group"


def not_tested_row(reason: str) -> dict:
    return {"comparison": "not tested", "test": "none",
            "statistic": np.nan, "df": np.nan, "p_value": np.nan,
            "p_adj_holm": np.nan, "note": reason}


def fit_groups(grouped: pd.DataFrame, groups_present: list[int]) -> dict:
    """只拟合，不写盘。"""
    fitters = {}
    for gid in groups_present:
        sub = grouped[grouped["group"] == gid]
        (kmf := KaplanMeierFitter(label=GROUP_LABELS[gid])).fit(
            sub["T"].to_numpy(), event_observed=sub["E"].to_numpy()
        )
        fitters[gid] = kmf
    return fitters


def compute_logrank(grouped: pd.DataFrame, groups_present: list[int]) -> list[dict]:
    """只计算，不写盘。少于两组时返回明确的 not tested 记录。"""
    mode = decide_test_mode(groups_present)
    if mode == "not_tested":
        return [not_tested_row(f"only {len(groups_present)} non-empty group(s); log-rank needs >= 2")]

    rows: list[dict] = []
    if mode == "two_group":
        a, b = groups_present[0], groups_present[1]
        r = logrank_test(
            grouped.loc[grouped["group"] == a, "T"],
            grouped.loc[grouped["group"] == b, "T"],
            event_observed_A=grouped.loc[grouped["group"] == a, "E"],
            event_observed_B=grouped.loc[grouped["group"] == b, "E"],
        )
        rows.append({"comparison": f"{GROUP_LABELS[a]} vs {GROUP_LABELS[b]}",
                     "test": "logrank", "statistic": float(r.test_statistic),
                     "df": 1, "p_value": float(r.p_value),
                     "p_adj_holm": float(r.p_value), "note": ""})
        return rows

    mv = multivariate_logrank_test(
        grouped["T"].to_numpy(), grouped["group"].to_numpy(), grouped["E"].to_numpy()
    )
    rows.append({"comparison": f"overall ({len(groups_present)} groups)",
                 "test": "multivariate_logrank",
                 "statistic": float(mv.test_statistic), "df": len(groups_present) - 1,
                 "p_value": float(mv.p_value), "p_adj_holm": float(mv.p_value), "note": ""})

    pairs = [(groups_present[i], groups_present[j])
             for i in range(len(groups_present)) for j in range(i + 1, len(groups_present))]
    raw_p, pair_rows = [], []
    for a, b in pairs:
        r = logrank_test(
            grouped.loc[grouped["group"] == a, "T"],
            grouped.loc[grouped["group"] == b, "T"],
            event_observed_A=grouped.loc[grouped["group"] == a, "E"],
            event_observed_B=grouped.loc[grouped["group"] == b, "E"],
        )
        raw_p.append(float(r.p_value))
        pair_rows.append({"comparison": f"{GROUP_LABELS[a]} vs {GROUP_LABELS[b]}",
                          "test": "logrank", "statistic": float(r.test_statistic),
                          "df": 1, "p_value": float(r.p_value), "note": ""})
    for row, pa in zip(pair_rows, holm(raw_p)):
        row["p_adj_holm"] = pa
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 绘图
def plot_overall(df: pd.DataFrame) -> tuple[KaplanMeierFitter, dict]:
    t = df["T"].to_numpy()
    e = df["E"].to_numpy()
    kmf = KaplanMeierFitter(label="All customers")
    kmf.fit(t, event_observed=e)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=False, linewidth=1.8)

    marks, annotations = {}, []
    for x in CHECKPOINTS:
        s, lo, hi = step_value(kmf, x)
        marks[x] = {"S": s, "lo": lo, "hi": hi, "at_risk": int((t >= x).sum())}
        if not np.isnan(s):
            ax.vlines(x, 0, s, colors="grey", linestyles=":", linewidth=1)
            ax.plot([x], [s], "o", color="black", markersize=4)
            annotations.append(f"{x}d: S={s:.3f}  [{lo:.3f}, {hi:.3f}]")
    ax.text(0.50, 0.86, "\n".join(annotations), transform=ax.transAxes,
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="grey", alpha=0.92))

    ax.set_xlabel("Days since first purchase")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Kaplan-Meier — churn after last purchase (30-day confirmation)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, min(760, int(t.max()) + 10))
    ax.grid(alpha=0.3)

    times = [0, 30, 60, 180, 365]
    ax.text(0.0, -0.19,
            "Number at risk:  " + "   ".join(f"{x}d={n}" for x, n in zip(times, at_risk(t, times))),
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(REPORTS, "km_overall.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"SAVED {out}")
    return kmf, marks


def plot_by_group(grouped: pd.DataFrame, groups_present: list[int], fitters: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for gid in groups_present:
        fitters[gid].plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.10,
                                            show_censors=False, linewidth=1.8, color=COLORS[gid])

    ax.set_xlabel("Days since first purchase")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Kaplan-Meier by first-invoice amount (earliest valid invoice)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 760)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    ann = []
    for x in CHECKPOINTS:
        parts = []
        for gid in groups_present:
            s, _, _ = step_value(fitters[gid], x)
            parts.append("NA" if np.isnan(s) else f"{s:.3f}")
        ann.append(f"{x:>3}d  " + "  ".join(parts))
    ax.text(0.50, 0.86,
            "S(t) at checkpoint (" + " / ".join(GROUP_LABELS[g].split(" ")[0] for g in groups_present) + ")\n" + "\n".join(ann),
            transform=ax.transAxes, fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="grey", alpha=0.92))

    times = [0, 30, 60, 180, 365]
    lines = []
    for gid in groups_present:
        sub = grouped[grouped["group"] == gid]
        lines.append(f"{GROUP_LABELS[gid]:16s} " + "  ".join(f"{x}d={n}" for x, n in zip(times, at_risk(sub["T"].to_numpy(), times))))
    ax.text(0.0, -0.24, "Number at risk\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=7.2, family="monospace")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = os.path.join(REPORTS, "km_by_group.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"SAVED {out}")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    if not os.path.exists(SURV):
        print(f"ERROR: {SURV} not found. Run src/build_survival.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(SURV, parse_dates=["entry_date", "last_purchase_date"])
    print(f"ROWS {len(df)}")
    problems = validate(df)
    if problems:
        for p in problems:
            print(f"QC_FAIL {p}", file=sys.stderr)
        print("ABORT: input failed QC.", file=sys.stderr)
        return 1
    print(f"QC_OK events={int(df['E'].sum())} censored={int((df['E'] == 0).sum())}")
    os.makedirs(REPORTS, exist_ok=True)

    # ================= 1. 总体：先交付，不依赖分组 =================
    kmf_all, marks_all = plot_overall(df)
    summary_rows = []
    for x in CHECKPOINTS:
        s, lo, hi = step_value(kmf_all, x)
        summary_rows.append({"group": "overall", "time_days": x,
                             "n_at_risk": int((df["T"] >= x).sum()),
                             "S_t": s, "CI_low": lo, "CI_high": hi})
    pd.DataFrame(summary_rows).to_csv(os.path.join(REPORTS, "km_summary.csv"), index=False)
    print("SAVED km_summary.csv (overall only)")
    print("OVERALL_DELIVERED")

    # ================= 2. 分组：先计算（无副作用）=================
    group_state = None
    group_error = None

    try:
        if not os.path.exists(RAW):
            raise FileNotFoundError(f"raw file missing: {RAW}")
        amt = load_first_invoice(set(df["user_id"].tolist()))
        merged = df.merge(amt, on="user_id", how="left", validate="one_to_one")
        n_missing = int(merged["first_invoice_amount"].isna().sum())
        grouped = merged.dropna(subset=["first_invoice_amount"]).copy()
        groups_present: list[int] = []
        if len(grouped):
            grouped["group"] = grouped["first_invoice_amount"].map(assign_group)
            groups_present = sorted(grouped["group"].unique().tolist())
        fitters = fit_groups(grouped, groups_present) if groups_present else {}
        lr_rows = compute_logrank(grouped, groups_present)      # 可能在这里抛错
        tied_users = int((amt["tied_time_invoices"] > 1).sum()) if len(amt) else 0
        group_state = {"amt": amt, "grouped": grouped, "groups_present": groups_present,
                       "fitters": fitters, "lr_rows": lr_rows,
                       "n_missing": n_missing, "tied_users": tied_users}
        print(f"GROUP_COMPUTED groups={groups_present} missing={n_missing} tied={tied_users}")
    except Exception as exc:  # noqa: BLE001
        group_error = f"{type(exc).__name__}: {exc}"
        group_state = None
        print(f"GROUP_FAILED {group_error}", file=sys.stderr)

    # ================= 3. 发布：全部成功才写分组产物 =================
    summary = pd.read_csv(os.path.join(REPORTS, "km_summary.csv"))
    published_groups = False

    if group_state is not None:
        g = group_state
        g["amt"].to_csv(INVOICE_AMOUNT, index=False)
        print(f"SAVED {INVOICE_AMOUNT} rows={len(g['amt'])}")

        if g["groups_present"]:
            plot_by_group(g["grouped"], g["groups_present"], g["fitters"])

            extra = []
            for gid in g["groups_present"]:
                sub = g["grouped"][g["grouped"]["group"] == gid]
                for x in CHECKPOINTS:
                    s, lo, hi = step_value(g["fitters"][gid], x)
                    extra.append({"group": GROUP_LABELS[gid], "time_days": x,
                                  "n_at_risk": int((sub["T"] >= x).sum()),
                                  "S_t": s, "CI_low": lo, "CI_high": hi})
            summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
            summary.to_csv(os.path.join(REPORTS, "km_summary.csv"), index=False)

        lr = pd.DataFrame(g["lr_rows"])
        lr.to_csv(os.path.join(REPORTS, "logrank.csv"), index=False)
        print(f"SAVED logrank.csv rows={len(lr)}")
        published_groups = True

        # 迁移表
        if os.path.exists(OLD_AMOUNT):
            try:
                old = pd.read_csv(OLD_AMOUNT)[["user_id", "first_order_amount"]]
                both = g["amt"][["user_id", "first_invoice_amount"]].merge(old, on="user_id", how="inner")
                both["old_group"] = both["first_order_amount"].map(assign_group)
                both["new_group"] = both["first_invoice_amount"].map(assign_group)
                migration = pd.crosstab(both["old_group"], both["new_group"],
                                        rownames=["old"], colnames=["new"])
                migration.to_csv(os.path.join(REPORTS, "group_migration.csv"))
                print(f"SAVED group_migration.csv (changed={int((both['old_group'] != both['new_group']).sum())})")
            except Exception as exc:  # noqa: BLE001
                print(f"MIGRATION_SKIPPED {exc}", file=sys.stderr)
    else:
        lr = pd.DataFrame([not_tested_row(group_error or "grouping unavailable")])
        lr.to_csv(os.path.join(REPORTS, "logrank.csv"), index=False)
        print(f"SAVED logrank.csv rows={len(lr)} (not tested)")

    # ================= 4. notes =================
    grp_counts = None
    if published_groups:
        grp_counts = group_state["grouped"].groupby("group").agg(
            n=("user_id", "size"), events=("E", "sum"),
            censored=("E", lambda s: int((s == 0).sum())),
            median_amount=("first_invoice_amount", "median"),
        )

    n_missing_text = f"{group_state['n_missing']}" if group_state is not None else "未评估（本次分组未完成）"
    tied_text = f"{group_state['tied_users']}" if group_state is not None else "未评估（本次分组未完成）"

    notes = ["# Kaplan-Meier 分析说明", ""]

    # 产物状态表
    status = [
        ("km_overall.png", "本次生成"),
        ("km_summary.csv", "本次生成（" + ("含分组" if published_groups else "仅总体") + "）"),
        ("logrank.csv", "本次生成" if published_groups else "本次生成（not tested）"),
    ]
    if published_groups:
        status.append(("km_by_group.png", "本次生成"))
        status.append(("group_migration.csv",
                       "本次生成" if os.path.exists(os.path.join(REPORTS, "group_migration.csv")) else "未生成"))
    else:
        status.append(("km_by_group.png", "**旧文件残留，不代表本次运行**"))
        status.append(("group_migration.csv", "**旧文件残留，不代表本次运行**"))

    notes += ["## 0. 本次产物状态", "", "| 文件 | 状态 |", "|---|---|"]
    for name, st in status:
        notes.append(f"| {BT}{name}{BT} | {st} |")
    notes += [""]

    if group_error:
        notes += [
            "> ⚠️ **本次分组分析未完成**：" + group_error,
            "> 分组相关的旧文件（图、迁移表）**不代表本次运行**；",
            "> 总体结果（第 2 节）仍然有效。", "",
        ]
    elif not published_groups:
        notes += ["> ⚠️ 分组已计算但未产出有效分组。", ""]

    notes += [
        "## 1. 输入与口径", "",
        f"- 输入：{BT}data/processed/survival.csv{BT}，{len(df):,} 行，**未被修改**",
        f"- 事件：**回溯末购事件**（{BT}E=1{BT}），时间回记至末购日，但要满 30 天才可确认",
        f"- 删失：**右删失**（{BT}E=0{BT}）",
        f"- 观察窗口：{OBS_START.date()} ~ {OBS_END.date()}（固定终点）",
        "", "## 2. 总体结果（不依赖分组）", "",
        "| 时点 | S(t) | 95% CI | 风险人数 |", "|---|---|---|---|",
    ]
    for x in CHECKPOINTS:
        m = marks_all[x]
        notes.append(f"| {x} 天 | {m['S']:.4f} | [{m['lo']:.4f}, {m['hi']:.4f}] | {m['at_risk']:,} |")

    notes += ["", "## 3. 分组定义", "",
              "| 组 | 规则 |", "|---|---|",
              f"| {GROUP_LABELS[0]} | 最早发票金额 <= {BINS[0]:.0f} GBP |",
              f"| {GROUP_LABELS[1]} | {BINS[0]:.0f} < 金额 <= {BINS[1]:.0f} GBP |",
              f"| {GROUP_LABELS[2]} | 金额 > {BINS[1]:.0f} GBP |", "",
              "- **口径**：同一客户、同一张发票内的有效明细（Quantity>0 且 Price>0）求和；",
              "  **只取该客户时间最早的一张发票**，同日后续发票不计入",
              "- **时间并列**：最早时间多于 1 张发票时，按发票号稳定排序取最小者",
              f"- **并列人数**：{tied_text}",
              f"- **缺失**：无首单金额的用户保留在总体曲线中，分组时排除（{n_missing_text}）",
              ""]

    if grp_counts is not None:
        notes += ["## 4. 各组样本", "",
                  "| 组 | 样本数 | 事件数 | 删失数 | 发票金额中位 |",
                  "|---|---|---|---|---|"]
        for gid in group_state["groups_present"]:
            r = grp_counts.loc[gid]
            notes.append(f"| {GROUP_LABELS[gid]} | {int(r['n']):,} | {int(r['events']):,} | {int(r['censored']):,} | {r['median_amount']:.2f} |")
        notes += ["", "## 5. log-rank", "",
                  "| 比较 | 检验 | 统计量 | 自由度 | p | Holm 校正 p |",
                  "|---|---|---|---|---|---|"]
        for row in group_state["lr_rows"]:
            if pd.isna(row["statistic"]):
                notes.append(f"| {row['comparison']} | {row['test']} | - | - | - | - |")
            else:
                notes.append(f"| {row['comparison']} | {row['test']} | {row['statistic']:.3f} | {row['df']} | {row['p_value']:.4g} | {row['p_adj_holm']:.4g} |")
        if any(row["test"] == "none" for row in group_state["lr_rows"]):
            notes += ["", "> 未做检验：" + group_state["lr_rows"][0]["note"]]
    else:
        notes += ["## 4. 分组", "", f"**未完成**：{group_error or 'grouping unavailable'}"]

    if published_groups and os.path.exists(os.path.join(REPORTS, "group_migration.csv")):
        mig = pd.read_csv(os.path.join(REPORTS, "group_migration.csv"), index_col=0)
        notes += ["", "## 6. 旧口径 → 新口径 迁移", "",
                  "旧口径 = 首购当日**所有**发票求和；新口径 = **最早一张发票**。",
                  "金额只会变小或不变，因此迁移只可能往低档走。", "",
                  "| 旧 \\ 新 | 0 | 1 | 2 |", "|---|---|---|---|"]
        for idx in mig.index:
            notes.append("| " + str(idx) + " | " + " | ".join(str(int(mig.loc[idx, c])) for c in mig.columns) + " |")
        changed = int(sum(int(mig.loc[i, c]) for i in mig.index for c in mig.columns if str(i) != str(c)))
        notes += ["", "共 **" + str(changed) + "** 人更换分组。", ""]

    notes += ["", "## 7. 必须说明的边界", "",
              "1. **曲线依赖固定观察终点**（2011-12-09）。终点之后无数据，末段估计不稳定。",
              "2. **事件时间回记至末购日**，但确认需要满 30 天；曲线在末购日附近的下降代表「事后确认为流失」，不是实时状态。",
              "3. **S(t) 不是「仍活跃率」，也不是付费概率**。",
              "4. **分组差异不代表因果**。首单金额是入组时的相关性特征。",
              "5. 本步骤**不做** Cox 回归、不做预测打分、不给出运营干预时点。",
              ""]

    with open(os.path.join(REPORTS, "km_notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    print("SAVED km_notes.md")

    print()
    print("=== CHECKPOINTS ===")
    for x in CHECKPOINTS:
        m = marks_all[x]
        print(f"  overall {x:>3}d  S={m['S']:.4f}  CI=[{m['lo']:.4f}, {m['hi']:.4f}]  at_risk={m['at_risk']}")
    if grp_counts is not None:
        print()
        print("=== GROUP COUNTS ===")
        print(grp_counts.to_string())
    print()
    print("=== LOGRANK ===")
    print(lr.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
