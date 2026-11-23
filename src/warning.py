"""Score feature snapshots using a portable, train-only Cox warning model."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 30
LAST_DAY = 60
LIMIT = 1000
TIERS = ("低", "中", "中高", "高")
FEATURE_LABELS = {
    "n_invoices": "购买发票数",
    "first_order_amount": "首单金额",
    "window_amount": "窗口消费金额",
    "days_since_last_buy": "距末购天数",
    "n_products": "商品种类数",
    "top1_product_share": "主购商品金额占比",
}
ACTION_MAP = {
    "days_since_last_buy": "推送：补货提醒",
    "n_invoices": "推送：复购入口",
    "first_order_amount": "试验候选：复购券",
    "window_amount": "试验候选：复购券",
    "n_products": "推送：相关商品",
    "top1_product_share": "推送：常购商品",
}
DEFAULT_ACTION = "推送：通用复购提醒"


def validate_snapshot(frame, features):
    unknown = set(features) - set(FEATURE_LABELS)
    if unknown:
        raise ValueError("模型包含未登记的特征：" + ", ".join(sorted(unknown)))
    required = ["user_id", *features]
    missing = set(required) - set(frame.columns)
    if missing or frame.empty or len(frame) > 50_000:
        raise ValueError("需要 1–50,000 行且包含字段：" + ", ".join(required))
    result = frame.loc[:, required].copy().reset_index(drop=True)
    ids = result["user_id"].astype("string").str.strip()
    if (ids.isna().any() or ids.eq("").any() or ids.duplicated().any()
            or ids.str.contains(r"^[=+@\-]|[\x00-\x1f]", regex=True, na=False).any()):
        raise ValueError("user_id 必须唯一、非空，且不能以公式字符开头或包含控制字符")
    result["user_id"] = ids
    numbers = result[features].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("所有特征必须是有限数值")
    # Rules apply only to columns the model actually uses; the whitelist can change on refit.
    present = set(features)
    count = numbers[[c for c in ("n_invoices", "n_products", "days_since_last_buy") if c in present]]
    amounts = numbers[[c for c in ("first_order_amount", "window_amount") if c in present]]
    if (count.lt(1).any().any() or count.mod(1).ne(0).any().any()
            or ("days_since_last_buy" in present and numbers["days_since_last_buy"].gt(14).any())
            or ("top1_product_share" in present and not numbers["top1_product_share"].between(0, 1).all())
            or amounts.le(0).any().any()
            or ({"first_order_amount", "window_amount"} <= present
                and numbers["first_order_amount"].gt(numbers["window_amount"] + 1e-8).any())):
        raise ValueError("检查计数、1–14 天购后间隔、商品占比及金额口径")
    result[features] = numbers
    return result


def score(frame, spec):
    features = spec["features"]
    data = validate_snapshot(frame, features)
    # Each positive contribution reduces log purchase hazard versus the training median.
    contribution = -(data[features] - pd.Series(spec["reference"])) * pd.Series(spec["coef"])
    eta = -contribution.sum(axis=1).to_numpy()
    if not np.isfinite(eta).all():
        raise ValueError("特征数值过大，无法产生有效预测")
    hazard = np.asarray(spec["reference_cumulative_hazard"])
    survival = np.exp(-np.outer(np.exp(eta.clip(-700, 700)), hazard))
    curves = pd.DataFrame(survival, columns=range(1, LAST_DAY + 1))
    risk = curves[DAY]
    ranked = np.argsort(-contribution.to_numpy(), axis=1, kind="stable")
    factors, actions = [], []
    for i, order in enumerate(ranked):
        active = [features[j] for j in order if contribution.iloc[i, j] > 0][:2]
        factors.append("；".join(
            FEATURE_LABELS[f] + ("偏高" if data.at[i, f] > spec["reference"][f] else "偏低")
            for f in active
        ) or "无正向模型贡献")
        actions.append(ACTION_MAP[active[0]] if active else DEFAULT_ACTION)
    scored = data[["user_id"]].assign(
        risk_score=risk,
        risk_tier=np.asarray(TIERS)[np.searchsorted(spec["thresholds"], risk, side="left")],
        risk_factors=factors,
        suggested_action=actions,
    )
    return scored, curves


def warning_list(scored):
    candidates = scored.loc[scored["risk_tier"].eq("高")].sort_values(
        ["risk_score", "user_id"], ascending=[False, True],
    ).head(LIMIT).copy()
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates.reset_index(drop=True)


def demo_snapshot():
    # Synthetic users, roughly matched to training-set quantiles (no real IDs or rows):
    # ~88% one invoice; first order median ~£290; products median ~19 (P90 ~56);
    # top-product share median ~0.13 (P90 ~0.47). Share is at least 1 / n_products.
    rng = np.random.default_rng(42)
    n = 240
    invoices = rng.choice([1, 2, 3, 4], n, p=[0.88, 0.10, 0.015, 0.005])
    first = rng.lognormal(5.65, 0.85, n).round(2)
    window = (first + (invoices - 1) * rng.lognormal(5.0, 0.7, n)).round(2)
    products = np.clip(rng.lognormal(2.9, 0.85, n).round(), 1, 200).astype(int)
    share = np.clip(rng.lognormal(-2.0, 0.9, n), 1 / products, 1.0).round(4)
    return pd.DataFrame({
        "user_id": [f"DEMO-{i:04d}" for i in range(1, n + 1)],
        "n_invoices": invoices,
        "first_order_amount": first,
        "window_amount": window,
        "days_since_last_buy": np.where(invoices == 1, 14, rng.integers(1, 14, n)),
        "n_products": products,
        "top1_product_share": share,
    })
