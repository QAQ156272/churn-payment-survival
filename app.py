"""Interactive warning demo. The bundled cohort contains no customer IDs."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import warning

ROOT = Path(__file__).resolve().parent
COLORS = ["#176780", "#8b5834", "#826f37", "#779489"]


@st.cache_data
def load_model():
    return json.loads((ROOT / "models" / "cox_warning.json").read_text(encoding="utf-8"))


st.set_page_config(page_title="复购预警台", layout="wide")
st.title("复购预警台")
st.caption("Cox B · 打分后 30 天未购买 · 分数用于排序，不代表永久流失")
spec = load_model()

left, right = st.columns([3, 2], vertical_alignment="bottom")
with left:
    source = st.radio("用户表来源", ["合成演示", "上传打分表"], horizontal=True)
with right:
    st.download_button("下载示例用户表", warning.demo_snapshot().head(10).to_csv(index=False).encode("utf-8-sig"),
                       file_name="demo_users.csv", mime="text/csv", icon=":material/download:")
try:
    if source == "上传打分表":
        uploaded = st.file_uploader("用户表 CSV", type=["csv"], max_upload_size=10)
        if uploaded is None:
            st.stop()
        snapshot = pd.read_csv(uploaded, dtype={"user_id": "string"}, keep_default_na=False, nrows=50_001)
    else:
        snapshot = warning.demo_snapshot()
    scored, curves = warning.score(snapshot, spec)
except (ValueError, UnicodeError) as exc:
    st.error(f"无法打分：{exc}")
    st.stop()

candidates = warning.warning_list(scored)
metrics = st.columns(3)
metrics[0].metric("已打分", f"{len(scored):,}")
metrics[1].metric("高风险候选", f"{len(candidates):,}")
metrics[2].metric("平均 30 天未购买分数", f"{scored['risk_score'].mean():.1%}")
st.divider()

tier_order = list(warning.TIERS[::-1])
palette = alt.Scale(domain=tier_order, range=COLORS)
chart_col, distribution_col = st.columns([3, 2])
with chart_col:
    st.subheader("未购买概率曲线")
    focus = st.selectbox("曲线范围", ["按风险档平均", "单个用户"])
    if focus == "单个用户":
        user = st.selectbox("用户 ID", scored["user_id"].tolist())
        idx = scored.index[scored["user_id"].eq(user)][0]
        plot = curves.loc[[idx]].T.rename(columns={idx: scored.at[idx, "risk_tier"]})
    else:
        plot = curves.groupby(scored["risk_tier"]).mean().T
    trend = plot.rename_axis(index="day", columns="tier").reset_index().melt(
        id_vars="day", value_name="survival",
    )
    curve = alt.Chart(trend).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=alt.X("day:Q", title="打分后第几天", scale=alt.Scale(domain=[1, 60])),
        y=alt.Y("survival:Q", title="未购买概率", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        color=alt.Color("tier:N", title="风险档", scale=palette),
        tooltip=[alt.Tooltip("day:Q", title="天数"), alt.Tooltip("tier:N", title="风险档"),
                 alt.Tooltip("survival:Q", title="预测未购买", format=".1%")],
    )
    guide = alt.Chart(pd.DataFrame({"day": [warning.DAY]})).mark_rule(
        color="#72828c", strokeDash=[3, 4], opacity=0.7,
    ).encode(x="day:Q")
    st.altair_chart((curve + guide).properties(height=255), width="stretch")

with distribution_col:
    st.subheader("风险分层分布")
    distribution = (scored["risk_tier"].value_counts().reindex(tier_order, fill_value=0)
                    .rename_axis("tier").reset_index(name="users"))
    bars = alt.Chart(distribution).mark_bar(size=26).encode(
        y=alt.Y("tier:N", sort=tier_order, title=None),
        x=alt.X("users:Q", title="人数"),
        color=alt.Color("tier:N", scale=palette, legend=None),
        tooltip=[alt.Tooltip("tier:N", title="风险档"), alt.Tooltip("users:Q", title="人数")],
    )
    st.altair_chart(bars.properties(height=250), width="stretch")

st.divider()
header, download = st.columns([3, 2], vertical_alignment="center")
with header:
    st.subheader("高风险预警名单")
with download:
    st.download_button("下载预警名单", candidates.to_csv(index=False).encode("utf-8-sig"),
                       file_name="warning_list.csv", mime="text/csv", disabled=candidates.empty,
                       icon=":material/download:", on_click="ignore")
if candidates.empty:
    st.info("当前用户表没有高风险候选。")
else:
    st.dataframe(
        candidates.assign(risk_score=candidates["risk_score"] * 100), hide_index=True,
        column_config={
            "rank": "排序", "user_id": "用户 ID", "risk_tier": "风险档", "risk_factors": "主要模型因素",
            "suggested_action": "建议动作",
            "risk_score": st.column_config.NumberColumn("未购买分数", format="%.1f%%"),
        },
        width="stretch",
    )
st.caption("动作仅为试验候选。高风险不等于干预有效；真实触达前须重查复购状态和触达许可。")
