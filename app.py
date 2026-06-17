import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "input"
if "result" not in st.session_state:
    st.session_state.result = None

# ── Global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div  { background: #0f0f0f !important; }
[data-testid="stSidebar"]            { display: none !important; }
[data-testid="stHeader"]             { background: #0f0f0f !important; border-bottom: 1px solid #1e1e1e; }
[data-testid="stDecoration"]         { display: none !important; }
.block-container                     { max-width: 780px !important; padding: 2rem 1.5rem 4rem !important; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #fff !important; }
p, label, div  { color: #d1d1d1; }

/* ── Top nav bar ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 2.5rem; padding-bottom: 1rem;
  border-bottom: 1px solid #1e1e1e;
}
.topbar-logo { font-size: 1.5rem; font-weight: 800; color: #E50914; letter-spacing: -0.5px; }
.topbar-sub  { font-size: .78rem; color: #666; margin-top: 1px; }
.step-badge  {
  font-size: .72rem; font-weight: 600; color: #888;
  background: #1a1a1a; border: 1px solid #2a2a2a;
  border-radius: 20px; padding: 5px 14px; letter-spacing: .5px;
}

/* ── Section labels ── */
.section-label {
  font-size: .7rem; font-weight: 700; color: #E50914;
  text-transform: uppercase; letter-spacing: 1.2px;
  margin: 2rem 0 .75rem;
}

/* ── Input card group ── */
.card {
  background: #161616; border: 1px solid #222;
  border-radius: 12px; padding: 1.5rem 1.5rem 0.5rem;
  margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSlider"] > div > div > div > div { background: #E50914 !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #555 !important; }

div[data-baseweb="select"] > div {
  background: #1e1e1e !important; border-color: #2e2e2e !important;
  border-radius: 8px !important; color: #fff !important;
}
div[data-baseweb="select"] * { color: #fff !important; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"]   input {
  background: #1e1e1e !important; border-color: #2e2e2e !important;
  border-radius: 8px !important; color: #fff !important;
}

label, [data-testid="stWidgetLabel"] p { color: #aaa !important; font-size: .85rem !important; }
[data-testid="stSlider"] label p       { color: #aaa !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
  background: #E50914 !important; color: #fff !important;
  border: none !important; border-radius: 9px !important;
  font-weight: 700 !important; font-size: 1rem !important;
  padding: 0.75rem 2rem !important; width: 100% !important;
  letter-spacing: .3px !important;
  box-shadow: 0 4px 24px rgba(229,9,20,.35) !important;
  transition: opacity .15s !important;
  cursor: pointer !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Secondary / ghost button ── */
.stButton > button[kind="secondary"] {
  background: transparent !important; color: #aaa !important;
  border: 1px solid #2e2e2e !important; border-radius: 9px !important;
  font-size: .88rem !important; padding: 0.55rem 1.5rem !important;
  width: auto !important; box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover { color: #fff !important; border-color: #555 !important; }

/* ── Result page ── */
.result-hero {
  border-radius: 16px; padding: 2.5rem 2rem 2rem;
  text-align: center; margin-bottom: 1.5rem;
  position: relative; overflow: hidden;
}
.result-hero.churn  { background: linear-gradient(145deg,#2a0000,#1a0000); border: 1px solid #6b0000; }
.result-hero.medium { background: linear-gradient(145deg,#2a1800,#1a1000); border: 1px solid #7a4500; }
.result-hero.safe   { background: linear-gradient(145deg,#001a0a,#000f06); border: 1px solid #0a5c28; }
.result-icon        { font-size: 3.5rem; margin-bottom: .5rem; }
.result-verdict     { font-size: 1.7rem; font-weight: 800; color: #fff; margin-bottom: .4rem; }
.result-prob        { font-size: 1rem; color: rgba(255,255,255,.7); margin-bottom: 1rem; }
.prob-pill {
  display: inline-block; border-radius: 30px;
  padding: 6px 22px; font-weight: 700; font-size: 1.05rem;
  color: #fff; margin-bottom: .5rem;
}
.prob-pill.churn  { background: rgba(229,9,20,.25); border: 1px solid #E50914; }
.prob-pill.medium { background: rgba(245,166,35,.2);  border: 1px solid #F5A623; }
.prob-pill.safe   { background: rgba(39,174,96,.2);   border: 1px solid #27AE60; }

/* ── Stat row ── */
.stat-row { display:flex; gap:12px; margin-bottom:1rem; flex-wrap:wrap; }
.stat-box {
  flex:1; min-width:120px;
  background:#161616; border:1px solid #222; border-radius:10px;
  padding:14px 16px; text-align:center;
}
.stat-label { font-size:.7rem; color:#666; text-transform:uppercase;
              letter-spacing:.8px; margin-bottom:5px; }
.stat-value { font-size:1.3rem; font-weight:700; color:#fff; }

/* ── Action card ── */
.action-card {
  background: #161616; border: 1px solid #222;
  border-left: 4px solid #E50914;
  border-radius: 12px; padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
}
.action-card.medium { border-left-color: #F5A623; }
.action-card.safe   { border-left-color: #27AE60; }
.action-title       { font-weight: 700; font-size: .95rem; color: #fff; margin-bottom: .3rem; }
.action-text        { font-size: .84rem; color: #999; line-height: 1.65; }

/* ── Summary table ── */
.summary-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px; margin-top: .75rem;
}
.summary-row {
  background: #161616; border: 1px solid #1e1e1e;
  border-radius: 8px; padding: 10px 14px;
  display: flex; justify-content: space-between; align-items: center;
}
.summary-key   { font-size: .78rem; color: #666; }
.summary-val   { font-size: .82rem; font-weight: 600; color: #ddd; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e1e1e; margin: 1.8rem 0; }

/* ── Progress bar ── */
.prob-bar-wrap { background:#1e1e1e; border-radius:99px; height:10px; margin:10px 0 6px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:99px; transition:width .5s; }

/* hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TRAIN MODEL  (cached, runs once per session)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def train_model():
    BASE = os.path.dirname(__file__)
    df   = pd.read_excel(os.path.join(BASE, "netflix_large_user_data.xlsx"))

    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[\s/()]+", "_", regex=True)
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
                  .str.strip("_"))

    PAY_COL    = "payment_history_ontime_delayed"
    INCOME_COL = "monthly_income"

    df["churn"]           = (df["churn_status_yes_no"].str.strip().str.lower() == "yes").astype(int)
    df["payment_delayed"] = (df[PAY_COL].str.strip().str.lower() == "delayed").astype(int)
    df["plan_encoded"]    = df["subscription_plan"].map({"Basic":1,"Standard":2,"Premium":3})

    sat_n = 1 - (df["customer_satisfaction_score_110"] - 1) / 9
    eng_n = 1 - (df["engagement_rate_110"] - 1) / 9
    sub_r = 1 - (df["subscription_length_months"] / 24)
    df["risk_score"] = (sat_n*.35 + eng_n*.30 + df["payment_delayed"]*.20
                        + sub_r.clip(0,1)*.15).round(4)

    feature_cols = [
        "subscription_length_months","customer_satisfaction_score_110",
        "daily_watch_time_hours","engagement_rate_110","support_queries_logged",
        "age", INCOME_COL,"promotional_offers_used","number_of_profiles_created",
        "payment_delayed","plan_encoded","risk_score",
        "subscription_plan","device_used_most_often","genre_preference","region",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    mdf      = df[feature_cols + ["churn"]].copy().dropna(subset=["churn"])
    cat_cols = mdf.select_dtypes(include=["object","string"]).columns.tolist()
    enc      = pd.get_dummies(mdf, columns=cat_cols, drop_first=True)
    bc       = enc.select_dtypes(bool).columns
    enc[bc]  = enc[bc].astype(int)

    X = enc.drop(columns=["churn"])
    y = enc["churn"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
    Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    sc = StandardScaler()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(sc.fit_transform(Xr), yr)

    yp = rf.predict(sc.transform(Xte)); ypr = rf.predict_proba(sc.transform(Xte))[:,1]

    cat_info = {col: sorted(df[col].dropna().unique().tolist())
                for col in [PAY_COL,"subscription_plan","device_used_most_often",
                            "genre_preference","region"] if col in df.columns}
    meta = {
        "accuracy": float(accuracy_score(yte,yp)), "precision": float(precision_score(yte,yp)),
        "recall":   float(recall_score(yte,yp)),   "f1":        float(f1_score(yte,yp)),
        "auc":      float(roc_auc_score(yte,ypr)),
        "total":    len(df), "churn_rate": float(df["churn"].mean()),
        "age_min":  int(df["age"].min()), "age_max": int(df["age"].max()),
        "inc_min":  int(df[INCOME_COL].min()), "inc_max": int(df[INCOME_COL].max()),
        "inc_mean": float(df[INCOME_COL].mean()),
        "pay_col":  PAY_COL, "cat_info": cat_info,
    }
    return rf, sc, X.columns.tolist(), meta

model, scaler, FEAT_NAMES, META = train_model()
CAT = META["cat_info"]

# ── Feature builder ────────────────────────────────────────────────────
def predict(inp):
    pay_del  = 1 if inp["payment"] == "Delayed" else 0
    plan_enc = {"Basic":1,"Standard":2,"Premium":3}.get(inp["plan"],1)
    sat_n    = 1 - (inp["satisfaction"]-1)/9
    eng_n    = 1 - (inp["engagement"]-1)/9
    risk_sc  = round(sat_n*.35 + eng_n*.30 + pay_del*.20
                     + max(0,min(1,1-inp["sub_len"]/24))*.15, 4)

    row = {
        "subscription_length_months":     inp["sub_len"],
        "customer_satisfaction_score_110": inp["satisfaction"],
        "daily_watch_time_hours":          inp["watch_time"],
        "engagement_rate_110":             inp["engagement"],
        "support_queries_logged":          inp["support_q"],
        "age":                             inp["age"],
        "monthly_income":                  inp["income"],
        "promotional_offers_used":         inp["promo"],
        "number_of_profiles_created":      inp["profiles"],
        "payment_delayed":                 pay_del,
        "plan_encoded":                    plan_enc,
        "risk_score":                      risk_sc,
    }
    dummies = {
        "subscription_plan":      ["Premium","Standard"],
        "device_used_most_often": ["Laptop","Mobile","Smart TV","Tablet"],
        "genre_preference":       ["Comedy","Documentary","Drama","Romance","Sci-Fi","Thriller"],
        "region":                 ["Asia","Europe","North America","South America"],
    }
    for col, cats in dummies.items():
        for cat in cats:
            row[f"{col}_{cat}"] = 1 if inp.get(col) == cat else 0

    df_r = pd.DataFrame([row])
    for c in FEAT_NAMES:
        if c not in df_r.columns: df_r[c] = 0
    df_r = df_r[FEAT_NAMES]

    prob = float(model.predict_proba(scaler.transform(df_r))[0][1])
    pred = int(model.predict(scaler.transform(df_r))[0])
    return prob, pred, risk_sc


# ══════════════════════════════════════════════════════════════════════
#  PAGE 1 — INPUT FORM
# ══════════════════════════════════════════════════════════════════════
def page_input():
    # Top bar
    st.markdown("""
    <div class="topbar">
      <div>
        <div class="topbar-logo">🎬 Netflix Churn Predictor</div>
        <div class="topbar-sub">Powered by Random Forest · JKUAT Data Science</div>
      </div>
      <div class="step-badge">Step 1 of 2 — Customer Details</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Enter customer information")
    st.markdown("<p style='color:#666;font-size:.9rem;margin-top:-8px;margin-bottom:1.5rem;'>"
                "Fill in the fields below and click <strong style='color:#fff'>Predict Churn</strong> "
                "to see the analysis.</p>", unsafe_allow_html=True)

    # ── Section: Who is this customer? ────────────────────────────────
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=META["age_min"],
                                  max_value=META["age_max"], value=32, step=1)
        with c2:
            income = st.number_input("Monthly Income ($)",
                                     min_value=META["inc_min"],
                                     max_value=META["inc_max"],
                                     value=int(META["inc_mean"]), step=100)
        with c3:
            region = st.selectbox("Region", CAT["region"])

    # ── Section: Subscription ─────────────────────────────────────────
    st.markdown('<div class="section-label">Subscription Details</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            plan    = st.selectbox("Plan", CAT["subscription_plan"])
            payment = st.selectbox("Payment History",
                                   ["On-Time" if "on" in v.lower() else "Delayed"
                                    for v in CAT[META["pay_col"]]])
            promo   = st.slider("Promotional Offers Used", 0, 10, 2)
        with c2:
            sub_len  = st.slider("Subscription Length (months)", 1, 24, 8)
            profiles = st.slider("Number of Profiles Created", 1, 5, 2)

    # ── Section: Viewing Behaviour ────────────────────────────────────
    st.markdown('<div class="section-label">Viewing Behaviour</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            watch_time   = st.slider("Daily Watch Time (hours)", 0.0, 5.0, 2.5, 0.1)
            satisfaction = st.slider("Satisfaction Score  (1 = very unhappy, 10 = very happy)", 1, 10, 7)
            support_q    = st.slider("Support Queries Logged", 0, 10, 1)
        with c2:
            engagement = st.slider("Engagement Rate  (1 = low, 10 = highly engaged)", 1, 10, 6)
            device     = st.selectbox("Primary Device", CAT["device_used_most_often"])
            genre      = st.selectbox("Favourite Genre", CAT["genre_preference"])

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Submit ─────────────────────────────────────────────────────────
    if st.button("Predict Churn  →", type="primary", use_container_width=True):
        st.session_state.inputs = dict(
            age=age, income=income, region=region,
            plan=plan, sub_len=sub_len, payment=payment,
            promo=promo, profiles=profiles, watch_time=watch_time,
            engagement=engagement, satisfaction=satisfaction,
            support_q=support_q, device_used_most_often=device,
            genre_preference=genre, subscription_plan=plan,
            col=region,
        )
        prob, pred, risk_sc = predict(st.session_state.inputs)
        st.session_state.result = dict(prob=prob, pred=pred, risk_sc=risk_sc,
                                       inputs=st.session_state.inputs)
        st.session_state.page = "result"
        st.rerun()

    # ── Footer note ────────────────────────────────────────────────────
    st.markdown(f"""
    <p style='text-align:center;color:#444;font-size:.75rem;margin-top:2rem;'>
      Model: Random Forest (Tuned) · AUC-ROC {META['auc']:.4f} · F1 {META['f1']:.4f}
      · Trained on {META['total']:,} customers
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE 2 — RESULT
# ══════════════════════════════════════════════════════════════════════
def page_result():
    r    = st.session_state.result
    prob = r["prob"]
    pred = r["pred"]
    rsc  = r["risk_sc"]
    inp  = r["inputs"]

    # Tier logic
    if prob >= .70:
        tier       = "High Risk"
        css_cls    = "churn"
        icon       = "⚠️"
        verdict    = "Likely to Churn"
        tier_color = "#E50914"
        action_hd  = "Retention Campaign A — Urgent"
        action_txt = ("This customer is showing strong churn signals. We recommend a personal "
                      "outreach call within 24 hours, offering an exclusive discount (e.g. 30% off "
                      "next 3 months) and a curated content recommendation based on their watch history.")
    elif prob >= .40:
        tier       = "Medium Risk"
        css_cls    = "medium"
        icon       = "🔔"
        verdict    = "Moderate Churn Risk"
        tier_color = "#F5A623"
        action_hd  = "Retention Campaign B — Proactive"
        action_txt = ("Moderate risk detected. Send a personalised email with a 'Top Picks For You' "
                      "carousel and a loyalty reward offer (e.g. a free premium month or partner "
                      "discount). Monitor engagement over the next 30 days.")
    else:
        tier       = "Low Risk"
        css_cls    = "safe"
        icon       = "✅"
        verdict    = "Likely to Stay"
        tier_color = "#27AE60"
        action_hd  = "No Intervention Needed"
        action_txt = ("This customer is engaged and satisfied. Continue with standard algorithmic "
                      "recommendations. Flag for re-evaluation if satisfaction or engagement scores "
                      "drop below 5 in the next review cycle.")

    pct = round(prob * 100, 1)

    # ── Top bar ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="topbar">
      <div>
        <div class="topbar-logo">🎬 Netflix Churn Predictor</div>
        <div class="topbar-sub">Powered by Random Forest · JKUAT Data Science</div>
      </div>
      <div class="step-badge">Step 2 of 2 — Prediction Result</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Back button ────────────────────────────────────────────────────
    if st.button("← Back to Customer Form", type="secondary"):
        st.session_state.page = "input"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result hero card ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-hero {css_cls}">
      <div class="result-icon">{icon}</div>
      <div class="result-verdict">{verdict}</div>
      <div class="result-prob">Churn probability score</div>
      <div class="prob-pill {css_cls}">{pct}%</div>
      <div class="prob-bar-wrap" style="max-width:320px;margin:14px auto 0;">
        <div class="prob-bar-fill"
             style="width:{pct}%;background:{tier_color};"></div>
      </div>
      <p style="color:#555;font-size:.76rem;margin-top:6px;">
        Risk Tier: <strong style="color:{tier_color};">{tier}</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Three stat boxes ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-label">Churn Probability</div>
        <div class="stat-value" style="color:{tier_color};">{pct}%</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Risk Score</div>
        <div class="stat-value">{rsc:.4f}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Model Decision</div>
        <div class="stat-value">{'Churn' if pred == 1 else 'Retain'}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Risk Tier</div>
        <div class="stat-value" style="color:{tier_color};font-size:1rem;">{tier}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommended action ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Recommended Action</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="action-card {css_cls}">
      <div class="action-title">{action_hd}</div>
      <div class="action-text">{action_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Churn Probability Gauge</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": "#fff", "size": 36}},
        gauge={
            "axis": {"range": [0,100], "tickcolor":"#444", "tickfont":{"color":"#555","size":11}},
            "bar":  {"color": tier_color, "thickness": .28},
            "bgcolor": "#161616", "bordercolor":"#2a2a2a",
            "steps": [
                {"range":[0,40],  "color":"#0d1f16"},
                {"range":[40,70], "color":"#1f1800"},
                {"range":[70,100],"color":"#1f0000"},
            ],
            "threshold": {"line":{"color":"#fff","width":2},
                          "thickness":.85,"value":prob*100},
        },
        title={"text":"Risk Level","font":{"color":"#555","size":13}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        height=260, margin=dict(l=20,r=20,t=40,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Input summary ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Customer Input Summary</div>', unsafe_allow_html=True)

    fields = [
        ("Age",              inp["age"]),
        ("Monthly Income",   f"${inp['income']:,}"),
        ("Region",           inp["region"]),
        ("Plan",             inp["plan"]),
        ("Subscription",     f"{inp['sub_len']} months"),
        ("Payment",          inp["payment"]),
        ("Watch Time",       f"{inp['watch_time']:.1f} hrs/day"),
        ("Engagement",       f"{inp['engagement']} / 10"),
        ("Satisfaction",     f"{inp['satisfaction']} / 10"),
        ("Support Queries",  inp["support_q"]),
        ("Device",           inp["device_used_most_often"]),
        ("Genre",            inp["genre_preference"]),
    ]

    html = '<div class="summary-grid">'
    for k, v in fields:
        html += f'<div class="summary-row"><span class="summary-key">{k}</span><span class="summary-val">{v}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Model info strip ───────────────────────────────────────────────
    st.markdown(f"""
    <hr class='divider'>
    <p style='text-align:center;color:#444;font-size:.75rem;'>
      Random Forest (Tuned) · AUC-ROC {META['auc']:.4f} · F1 {META['f1']:.4f}
      · Accuracy {META['accuracy']:.1%} · Trained on {META['total']:,} customers
    </p>
    """, unsafe_allow_html=True)

    # ── Predict another ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Another Customer  →", type="primary", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════
if st.session_state.page == "input":
    page_input()
else:
    page_result()
