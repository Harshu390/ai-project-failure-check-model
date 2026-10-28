from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Project Failure-Risk Scorer", page_icon="📉", layout="wide")
BASE = Path(__file__).resolve().parent

FEATS = ["data_readiness", "leadership_sponsorship", "team_ai_maturity", "integration_complexity",
         "engineer_buy_in", "team_capacity", "use_case_fit", "poc_before_purchase", "scope_narrow"]

# key: (label, rubric text shown on hover, input kind)
FACTORS = {
    "data_readiness": ("Data readiness",
        "Low: messy, untagged, big gaps | Medium: usable after cleanup | High: clean, tagged, accessible", "level"),
    "leadership_sponsorship": ("Leadership sponsorship",
        "Low: nobody owns it | Medium: approved but passive | High: named executive sponsor with budget", "level"),
    "team_ai_maturity": ("Team AI / ML maturity",
        "Low: no ML experience | Medium: some experience | High: has shipped ML to production before", "level"),
    "integration_complexity": ("Integration complexity",
        "HIGHER = MORE COMPLEX = RISKIER. Low: plugs in easily | Medium: some custom work | High: deep legacy integration", "level"),
    "engineer_buy_in": ("Engineer / end-user buy-in",
        "Low: team opposes or ignores it | Medium: neutral | High: the team asked for it", "level"),
    "team_capacity": ("Team capacity to implement",
        "Low: already overloaded | Medium: some spare time | High: dedicated people assigned", "level"),
    "use_case_fit": ("Use-case fit for AI",
        "Poor: simple rules would do the job | Medium: mixed | Good: genuinely needs ML / AI", "fit"),
    "poc_before_purchase": ("Proof-of-concept before purchase?",
        "Yes if a time-boxed trial or sandbox test ran before committing budget", "yesno"),
    "scope_narrow": ("Narrow, well-scoped use case?",
        "Yes = one focused problem. No = broad 'do everything' ambition", "yesno"),
}
LEVELS = {"Low": 1, "Medium": 2, "High": 3}
FIT = {"Poor": 1, "Medium": 2, "Good": 3}
YN = {"No": 0, "Yes": 1}
BEST = {"data_readiness": 3, "leadership_sponsorship": 3, "team_ai_maturity": 3, "integration_complexity": 1,
        "engineer_buy_in": 3, "team_capacity": 3, "use_case_fit": 3, "poc_before_purchase": 1, "scope_narrow": 1}


@st.cache_resource
def load_model():
    df = pd.read_csv(BASE / "dataset.csv")
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=1000, C=0.3))]).fit(df[FEATS], df["failed"])


model = load_model()


def text_of(key, val):
    kind = FACTORS[key][2]
    m = LEVELS if kind == "level" else FIT if kind == "fit" else YN
    return {v: k for k, v in m.items()}[val]


def one_input(container, key, prefix):
    label, help_, kind = FACTORS[key]
    k = f"{prefix}_{key}"
    if kind == "level":
        return LEVELS[container.selectbox(label, list(LEVELS), index=1, help=help_, key=k)]
    if kind == "fit":
        return FIT[container.selectbox(label, list(FIT), index=1, help=help_, key=k)]
    return YN[container.radio(label, list(YN), index=0, horizontal=True, help=help_, key=k)]


def risk(vals):
    return float(model.predict_proba(pd.DataFrame([vals])[FEATS])[0, 1])


def decision(p):
    if p < 0.30:
        return "GO", "#2E8B57"
    if p < 0.60:
        return "GO WITH CONDITIONS", "#E6A100"
    return "HOLD - fix gaps first", "#C0392B"


def gap_table(vals):
    base = risk(vals)
    rows = []
    for k in FEATS:
        if vals[k] == BEST[k]:
            continue
        v2 = dict(vals); v2[k] = BEST[k]
        gain = base - risk(v2)
        if gain > 0.005:
            rows.append((k, FACTORS[k][0], text_of(k, vals[k]), text_of(k, BEST[k]), gain * 100))
    rows.sort(key=lambda r: -r[4])
    return rows


def fix_top3(vals, rows):
    v2 = dict(vals)
    for r in rows[:3]:
        v2[r[0]] = BEST[r[0]]
    return risk(v2)


st.title("AI Project Failure-Risk Scorer")
st.caption(
    "Pre-investment readiness check. Built from 17 coded real case studies (mostly IT-operations / AIOps, plus some "
    "general AI pilots) and 63 labelled synthetic cases. The nine factors are domain-neutral, so any AI or analytics "
    "project can be scored, but accuracy outside IT operations is untested. A prototype for structuring a readiness "
    "discussion, not a validated predictor.")

tab1, tab2 = st.tabs(["Score one project", "Compare projects"])

# ------------------------------------------------------------------ one project
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Rate the project")
        c1, c2 = st.columns(2)
        vals = {}
        for i, k in enumerate(FACTORS):
            vals[k] = one_input(c1 if i < 5 else c2, k, "one")
    p = risk(vals)
    label, color = decision(p)
    rows = gap_table(vals)
    with right:
        st.subheader("Result")
        st.markdown(f"<h1 style='color:{color};margin-bottom:0'>{p*100:.0f}% failure risk</h1>"
                    f"<h3 style='color:{color};margin-top:0'>{label}</h3>", unsafe_allow_html=True)
        st.caption("Bands: under 30% go, 30-60% go with conditions, over 60% hold until the gaps are fixed.")
        if rows:
            p3 = fix_top3(vals, rows)
            st.markdown(f"**If you fix the top {min(3, len(rows))} gaps together: {p*100:.0f}% -> {p3*100:.0f}%** "
                        f"({decision(p3)[0]})")
            st.dataframe(pd.DataFrame(
                [(r[1], r[2], r[3], f"-{r[4]:.0f} pts") for r in rows],
                columns=["Gap to close", "Now", "Target", "Risk reduction (alone)"]),
                hide_index=True)
            fig, ax = plt.subplots(figsize=(5, 2.6))
            ax.barh([r[1] for r in rows][::-1], [r[4] for r in rows][::-1], color="#2E75B6")
            ax.set_xlabel("Risk reduction if this gap alone is closed (points)")
            plt.tight_layout(); st.pyplot(fig)
        else:
            st.success("Every factor is already at its best level.")

# ------------------------------------------------------------------ compare
with tab2:
    st.subheader("Compare candidate projects")
    n = st.slider("How many projects?", 2, 4, 3)
    cols = st.columns(n)
    results = []
    for i, col in enumerate(cols):
        name = col.text_input("Project name", value=f"Project {chr(65 + i)}", key=f"cmp_name_{i}")
        ptype = col.text_input("Type (e.g. hospital billing AI)", value="", key=f"cmp_type_{i}")
        pv = {k: one_input(col, k, f"cmp{i}") for k in FACTORS}
        pr = risk(pv)
        g = gap_table(pv)
        results.append({"Project": name, "Type": ptype, "Failure risk %": round(pr * 100),
                        "Decision": decision(pr)[0],
                        "Top gaps to fix": ", ".join(r[1] for r in g[:3]) or "none",
                        "_c": decision(pr)[1]})
    res = pd.DataFrame(results).sort_values("Failure risk %").reset_index(drop=True)
    st.markdown("### Ranking (lowest risk first)")
    st.dataframe(res.drop(columns="_c"), hide_index=True)
    fig, ax = plt.subplots(figsize=(7, 0.8 + 0.6 * len(res)))
    ax.barh(res["Project"][::-1], res["Failure risk %"][::-1], color=list(res["_c"][::-1]))
    ax.set_xlim(0, 105); ax.set_xlabel("Failure risk %")
    for y_, v_ in enumerate(res["Failure risk %"][::-1]):
        ax.text(v_ + 1, y_, f"{v_}%", va="center")
    plt.tight_layout(); st.pyplot(fig)
    st.download_button("Download comparison as CSV", res.drop(columns="_c").to_csv(index=False),
                       file_name="project_comparison.csv", mime="text/csv")