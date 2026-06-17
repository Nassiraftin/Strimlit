import streamlit as st
import numpy as np
import pandas as pd
import joblib, json, os
import plotly.graph_objects as go

# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Netflix dark theme ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] { background:#141414 !important; color:#fff; }
[data-testid="stSidebar"]  { background:#000 !important; }
[data-testid="stSidebar"] * { color:#fff !important; }
[data-testid="stHeader"]   { background:#000 !important; }
.block-container { padding-top:1.5rem !important; }

/* ── Hero banner ── */
.hero {
  background:linear-gradient(135deg,#E50914 0%,#831010 100%);
  border-radius:14px; padding:28px 36px; margin-bottom:22px;
  box-shadow:0 8px 40px rgba(229,9,20,.4);
  display:flex; align-items:center; gap:22px;
}
.hero-logo { font-size:3.6rem; line-height:1; }
.hero-title { font-size:2rem; font-weight:900; color:#fff; margin:0 0 4px; letter-spacing:-.5px; }
.hero-sub   { font-size:.95rem; color:rgba(255,255,255,.82); margin:0; }

/* ── KPI cards ── */
.kpi-wrap { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }
.kpi {
  flex:1; min-width:130px;
  background:#1a1a1a; border:1px solid #2a2a2a;
  border-radius:10px; padding:16px 18px;
  text-align:center;
}
.kpi-label { font-size:.72rem; color:#888; text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
.kpi-value { font-size:1.6rem; font-weight:800; }

/* ── Result cards ── */
.res-card {
  border-radius:14px; padding:28px 24px;
  text-align:center; margin-bottom:16px;
}
.res-churn  { background:linear-gradient(135deg,#E50914,#7a0000);
              box-shadow:0 6px 32px rgba(229,9,20,.5); }
.res-medium { background:linear-gradient(135deg,#F5A623,#a86e00);
              box-shadow:0 6px 32px rgba(245,166,35,.4); }
.res-safe   { background:linear-gradient(135deg,#27AE60,#145c30);
              box-shadow:0 6px 32px rgba(39,174,96,.4); }
.res-icon  { font-size:3.2rem; margin-bottom:6px; }
.res-title { font-size:1.6rem; font-weight:900; color:#fff; margin-bottom:4px; }
.res-prob  { font-size:1.05rem; color:rgba(255,255,255,.88); }
.res-tier  { font-size:.9rem; color:rgba(255,255,255,.75); margin-top:6px; font-weight:700; }

/* ── Section header ── */
.sec-hdr {
  font-size:.82rem; font-weight:700; color:#E50914;
  border-left:4px solid #E50914; padding-left:10px;
  text-transform:uppercase; letter-spacing:.9px;
  margin:18px 0 10px;
}

/* ── Intervention box ── */
.intv {
  background:#1a1a1a; border:1px solid #333;
  border-left:5px solid #E50914; border-radius:8px;
  padding:14px 18px; margin-top:14px;
}
.intv-title { font-weight:800; color:#E50914; font-size:.88rem; margin-bottom:5px; }
.intv-text  { color:#ccc; font-size:.84rem; line-height:1.6; }

/* ── Predict button ── */
.stButton > button {
  background:linear-gradient(135deg,#E50914,#a00000) !important;
  color:#fff !important; border:none !important;
  font-size:1rem !important; font-weight:800 !important;
  letter-spacing:.8px !important;
  border-radius:8px !important; padding:13px !important;
  width:100% !important;
  box-shadow:0 4px 18px rgba(229,9,20,.45) !important;
  transition:all .2s !important;
}
.stButton > button:hover { transform:translateY(-2px) !important;
  box-shadow:0 8px 28px rgba(229,9,20,.65) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#000; border-bottom:1px solid #2a2a2a; }
.stTabs [data-baseweb="tab"]      { color:#888 !important; }
.stTabs [aria-selected="true"]    { color:#E50914 !important; border-bottom:2px solid #E50914 !important; }

/* ── Misc ── */
div[data-testid="metric-container"] > div { background:#1a1a1a; border-radius:8px; padding:10px 14px; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load artefacts ────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artefacts():
    model   = joblib.load(os.path.join(BASE, "netflix_churn_rf_model.pkl"))
    scaler  = joblib.load(os.path.join(BASE, "netflix_churn_scaler.pkl"))
    feat_names = joblib.load(os.path.join(BASE, "netflix_feature_names.pkl"))
    with open(os.path.join(BASE, "netflix_app_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, feat_names, meta

model, scaler, FEAT_NAMES, META = load_artefacts()
CAT_INFO = META["cat_info"]

# ─── Constants ────────────────────────────────────────────────────────
PLAN_ORDER = {"Basic":1,"Standard":2,"Premium":3}
TIERS = {
    "High Risk":   ("#E50914","res-churn","⚠️"),
    "Medium Risk": ("#F5A623","res-medium","🔔"),
    "Low Risk":    ("#27AE60","res-safe","✅"),
}
INTERVENTIONS = {
    "High Risk":   ("🚨 Retention Campaign A",
                    "Immediate personal outreach + exclusive discount offer + "
                    "hand-picked content recommendation based on viewing history."),
    "Medium Risk": ("📧 Retention Campaign B",
                    "Targeted email with 'Top Picks For You' carousel + "
                    "loyalty reward prompt (e.g. free premium month)."),
    "Low Risk":    ("✅ Standard Experience",
                    "Regular algorithmic recommendations — no special intervention required."),
}

def assign_tier(prob):
    if prob >= .70: return "High Risk"
    if prob >= .40: return "Medium Risk"
    return "Low Risk"

# ─── Feature builder ───────────────────────────────────────────────────
def build_row(inp: dict) -> pd.DataFrame:
    pay_del  = 1 if inp["payment"] == "Delayed" else 0
    plan_enc = PLAN_ORDER.get(inp["plan"], 1)
    sat_n    = 1 - (inp["satisfaction"] - 1) / 9
    eng_n    = 1 - (inp["engagement"]   - 1) / 9
    sub_risk = max(0, min(1, 1 - inp["sub_len"] / 24))
    risk_sc  = round(sat_n*.35 + eng_n*.30 + pay_del*.20 + sub_risk*.15, 4)

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

    # One-hot (drop_first=True → alphabetically first category dropped)
    dummies = {
        "subscription_plan":      ("Basic",    ["Premium","Standard"]),
        "device_used_most_often": ("Desktop",  ["Laptop","Mobile","Smart TV","Tablet"]),
        "genre_preference":       ("Action",   ["Comedy","Documentary","Drama","Romance","Sci-Fi","Thriller"]),
        "region":                 ("Africa",   ["Asia","Europe","North America","South America"]),
    }
    for col, (dropped, cats) in dummies.items():
        for cat in cats:
            row[f"{col}_{cat}"] = 1 if inp.get(col) == cat else 0

    df = pd.DataFrame([row])
    # Ensure exact column order expected by model
    for c in FEAT_NAMES:
        if c not in df.columns:
            df[c] = 0
    return df[FEAT_NAMES], risk_sc

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Netflix Churn Predictor")
    st.markdown("---")
    st.success("✅ Model loaded — Random Forest (Tuned) ★")
    st.markdown(f"**Dataset:** {META['total_customers']:,} customers · "
                f"Churn rate: {META['churn_rate']*100:.1f}%")
    st.markdown("---")

    st.markdown('<p class="sec-hdr">👤 Customer Profile</p>', unsafe_allow_html=True)
    age    = st.slider("Age", META["age_min"], META["age_max"], 32)
    income = st.slider("Monthly Income ($)", META["income_min"], META["income_max"],
                       int(META["income_mean"]), step=100)
    region = st.selectbox("Region", CAT_INFO["region"])

    st.markdown("---")
    st.markdown('<p class="sec-hdr">📋 Subscription</p>', unsafe_allow_html=True)
    plan    = st.selectbox("Subscription Plan", CAT_INFO["subscription_plan"])
    sub_len = st.slider("Subscription Length (months)", 1, 24, 8)
    payment = st.selectbox("Payment History",
                           ["On-Time" if v=="On-Time" else "Delayed"
                            for v in CAT_INFO["payment_history_ontime_delayed"]])
    promo    = st.slider("Promotional Offers Used", 0, 10, 2)
    profiles = st.slider("Number of Profiles", 1, 5, 2)

    st.markdown("---")
    st.markdown('<p class="sec-hdr">📺 Engagement & Content</p>', unsafe_allow_html=True)
    watch_time   = st.slider("Daily Watch Time (hrs)", 0.0, 5.0, 2.5, 0.1)
    engagement   = st.slider("Engagement Rate (1–10)", 1, 10, 6)
    satisfaction = st.slider("Satisfaction Score (1–10)", 1, 10, 7)
    support_q    = st.slider("Support Queries Logged", 0, 10, 1)
    device       = st.selectbox("Device Used Most", CAT_INFO["device_used_most_often"])
    genre        = st.selectbox("Genre Preference", CAT_INFO["genre_preference"])

    st.markdown("---")
    predict_btn = st.button("🔍  PREDICT CHURN RISK", use_container_width=True)

# Bundle inputs
inputs = dict(age=age, income=income, region=region, plan=plan, sub_len=sub_len,
              payment=payment, promo=promo, profiles=profiles,
              watch_time=watch_time, engagement=engagement,
              satisfaction=satisfaction, support_q=support_q,
              device_used_most_often=device, genre_preference=genre,
              subscription_plan=plan)

# ══════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-logo">🎬</div>
  <div>
    <p class="hero-title">Netflix Customer Churn Predictor</p>
    <p class="hero-sub">
      Random Forest (Tuned) ★ &nbsp;|&nbsp;
      AUC-ROC: {auc:.4f} &nbsp;|&nbsp;
      F1-Score: {f1:.4f} &nbsp;|&nbsp;
      Accuracy: {acc:.1%}
    </p>
  </div>
</div>
""".format(**META), unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────
kpis = [
    ("🧠 Model",      "Random Forest",   "#E50914"),
    ("🎯 Accuracy",   f"{META['accuracy']:.1%}", "#F5A623"),
    ("📊 AUC-ROC",    f"{META['auc']:.4f}",      "#4A90D9"),
    ("📈 F1-Score",   f"{META['f1']:.4f}",        "#27AE60"),
    ("👥 Customers",  f"{META['total_customers']:,}", "#9B59B6"),
    ("🔄 Churn Rate", f"{META['churn_rate']*100:.1f}%", "#E50914"),
]
cols = st.columns(len(kpis))
for col, (label, val, colour) in zip(cols, kpis):
    col.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{colour};">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════
feat_df, risk_sc = build_row(inputs)
scaled           = scaler.transform(feat_df)
prob             = float(model.predict_proba(scaled)[0][1])
pred             = int(model.predict(scaled)[0])
tier             = assign_tier(prob)
tier_colour, res_cls, tier_icon = TIERS[tier]
int_title, int_text             = INTERVENTIONS[tier]

left, right = st.columns([1, 1], gap="large")

# ── Left: result + intervention ────────────────────────────────────────
with left:
    st.markdown('<p class="sec-hdr">🎯 Prediction Result</p>', unsafe_allow_html=True)

    if pred == 1:
        label = "LIKELY TO CHURN"
    elif tier == "Medium Risk":
        label = "MODERATE CHURN RISK"
    else:
        label = "LIKELY TO STAY"

    st.markdown(f"""
    <div class="res-card {res_cls}">
      <div class="res-icon">{tier_icon}</div>
      <div class="res-title">{label}</div>
      <div class="res-prob">Churn Probability: <strong>{prob:.1%}</strong></div>
      <div class="res-tier">Risk Tier: {tier}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{prob:.3f}")
    m2.metric("Risk Score",        f"{risk_sc:.4f}")
    m3.metric("Prediction",        "Churn" if pred else "Retain")

    # Intervention
    st.markdown(f"""
    <div class="intv">
      <div class="intv-title">{int_title}</div>
      <div class="intv-text">{int_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Risk tier table
    st.markdown('<p class="sec-hdr">📊 Risk Tier Reference</p>', unsafe_allow_html=True)
    tier_df = pd.DataFrame({
        "Tier": ["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
        "Probability": ["≥ 70%", "40% – 69%", "< 40%"],
        "Action": ["Campaign A – Personal outreach",
                   "Campaign B – Email + loyalty reward",
                   "Standard experience"],
    })
    st.dataframe(tier_df, hide_index=True, use_container_width=True)

# ── Right: gauge + feature bar ─────────────────────────────────────────
with right:
    st.markdown('<p class="sec-hdr">📈 Churn Probability Gauge</p>', unsafe_allow_html=True)

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix":"%","font":{"color":"#fff","size":42}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#666","tickfont":{"color":"#aaa"}},
            "bar": {"color":tier_colour,"thickness":.3},
            "bgcolor":"#1a1a1a","bordercolor":"#333",
            "steps":[
                {"range":[0,40],  "color":"#0a2018"},
                {"range":[40,70], "color":"#221800"},
                {"range":[70,100],"color":"#1f0000"},
            ],
            "threshold":{"line":{"color":"#fff","width":3},
                         "thickness":.85,"value":prob*100},
        },
        title={"text":"Churn Risk Level","font":{"color":"#fff","size":15}},
    ))
    fig_g.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414",
                        height=290, margin=dict(l=30,r=30,t=50,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    # Key input snapshot bar
    st.markdown('<p class="sec-hdr">🔑 Input Snapshot (normalised)</p>', unsafe_allow_html=True)
    snap = {
        "Satisfaction":  satisfaction / 10,
        "Engagement":    engagement   / 10,
        "Risk Score":    risk_sc,
        "Watch Time":    watch_time   / 5,
        "Support Q's":   min(support_q / 10, 1),
        "Sub Length":    min(sub_len   / 24, 1),
    }
    fig_b = go.Figure(go.Bar(
        x=list(snap.values()), y=list(snap.keys()), orientation="h",
        marker_color=["#27AE60","#4A90D9","#E50914","#F5A623","#9B59B6","#7F8C8D"],
        text=[f"{v:.2f}" for v in snap.values()],
        textposition="outside", textfont={"color":"#fff"},
    ))
    fig_b.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
        font_color="#fff", height=230,
        xaxis=dict(range=[0,1.35],gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        margin=dict(l=10,r=40,t=8,b=8),
    )
    st.plotly_chart(fig_b, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊  Model Performance", "🔍  Feature Importance", "📖  How to Use"])

with tab1:
    st.markdown("### All-Model Comparison")
    perf = pd.DataFrame({
        "Model":     ["Decision Tree","Logistic Regression",
                      "Random Forest (Default)","Random Forest (Tuned) ★"],
        "Accuracy":  [.4567,.5200,.5033,.4933],
        "Precision": [.4966,.5517,.5376,.5291],
        "Recall":    [.4444,.5926,.5741,.5617],
        "F1-Score":  [.4691,.5714,.5552,.5449],
        "AUC-ROC":   [.4280,.5012,.5109,.5337],
    })
    metrics   = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
    pal_model = ["#7F8C8D","#4A90D9","#F5A623","#E50914"]

    fig_cmp = go.Figure()
    for i, row in perf.iterrows():
        fig_cmp.add_trace(go.Bar(
            name=row["Model"], x=metrics,
            y=[row[m] for m in metrics],
            marker_color=pal_model[i],
            text=[f"{row[m]:.3f}" for m in metrics],
            textposition="outside", textfont={"color":"#fff","size":9},
        ))
    fig_cmp.update_layout(
        barmode="group", paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
        font_color="#fff", height=400,
        legend=dict(bgcolor="#1a1a1a",bordercolor="#444"),
        yaxis=dict(range=[0,1.15],gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        title=dict(text="Model Performance Comparison — Netflix Churn Prediction",
                   font={"color":"#fff","size":14}),
        margin=dict(l=10,r=10,t=50,b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    styled = perf.set_index("Model").style.background_gradient(
        cmap="RdYlGn", subset=metrics, vmin=.4, vmax=.65
    ).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

with tab2:
    st.markdown("### Feature Importances — Random Forest (Tuned) ★")
    importances = model.feature_importances_
    fi = (pd.DataFrame({"Feature": FEAT_NAMES, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(20))

    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker_color="#E50914", marker_line_color="#fff", marker_line_width=.5,
        text=[f"{v:.4f}" for v in fi["Importance"]],
        textposition="outside", textfont={"color":"#fff","size":9},
    ))
    fig_fi.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
        font_color="#fff", height=540,
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        title=dict(text="Top Feature Importances (Mean Decrease in Gini Impurity)",
                   font={"color":"#fff"}),
        margin=dict(l=10,r=80,t=50,b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with tab3:
    st.markdown("""
### How to Use This App

**Step 1 — Fill the sidebar**

| Section | Fields |
|---------|--------|
| 👤 Customer Profile | Age, monthly income, region |
| 📋 Subscription | Plan, length (months), payment history, promo offers, profiles |
| 📺 Engagement | Daily watch time, engagement & satisfaction (1–10), support queries, device, genre |

**Step 2 — Click "PREDICT CHURN RISK"**

The app automatically:
- Computes the **composite risk score** (satisfaction × 35% + engagement × 30% + payment × 20% + tenure × 15%)
- One-hot encodes categorical inputs to match training features
- Scales with the fitted `StandardScaler`
- Returns a **churn probability** from the tuned Random Forest

**Step 3 — Act on the result**

| 🏷️ Tier | Probability | Strategy |
|---------|-------------|---------|
| 🔴 High Risk | ≥ 70% | Personal outreach + immediate discount |
| 🟡 Medium Risk | 40–69% | Targeted email + loyalty reward |
| 🟢 Low Risk | < 40% | Standard experience |

---

### About the Model
- **Algorithm:** Random Forest Classifier (100 trees, Gini criterion)
- **Preprocessing:** SMOTE oversampling → StandardScaler
- **Split:** 70/30 stratified train/test
- **Dataset:** 1,000 Netflix subscribers, 16 raw features → 28 engineered features
- **Best metric:** AUC-ROC 0.5337 (selected over Logistic Regression for deployment)
""")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#444;font-size:.76rem;
  margin-top:40px;border-top:1px solid #222;padding-top:14px;">
  Netflix Customer Churn Prediction &nbsp;|&nbsp;
  Random Forest (Tuned) ★ &nbsp;|&nbsp;
  Streamlit + Plotly &nbsp;|&nbsp;
  JKUAT Data Science &amp; Analytics — SCT213-C002-0094/2022
</div>
""", unsafe_allow_html=True)
