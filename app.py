import streamlit as st

from churn_core import train_model, base_css

st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(base_css(), unsafe_allow_html=True)

model, scaler, FEAT_NAMES, META = train_model()
CAT_INFO = META["cat_info"]

st.markdown(
    f"""
    <div class="hero">
        <p class="hero-title">Netflix Customer Churn Predictor</p>
        <p class="hero-sub">
            Enter a customer's profile below and we'll estimate how likely
            they are to cancel their subscription. Model trained on
            {META['total_customers']:,} subscriber records.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="sec-hdr">Customer profile</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.slider("Age", META["age_min"], META["age_max"], 32)
with c2:
    income = st.slider(
        "Monthly income ($)", META["income_min"], META["income_max"],
        int(META["income_mean"]), step=100,
    )
with c3:
    region = st.selectbox("Region", CAT_INFO["region"])

st.markdown('<p class="sec-hdr">Subscription</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    plan = st.selectbox("Plan", CAT_INFO["subscription_plan"])
with c2:
    sub_len = st.slider("Length (months)", 1, 24, 8)
with c3:
    payment = st.selectbox(
        "Payment history",
        ["On-Time" if "on" in v.lower() else "Delayed" for v in CAT_INFO[META["pay_col"]]],
    )
with c4:
    profiles = st.slider("Profiles on account", 1, 5, 2)

promo = st.slider("Promotional offers used", 0, 10, 2)

st.markdown('<p class="sec-hdr">Viewing &amp; engagement</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    watch_time = st.slider("Daily watch time (hours)", 0.0, 5.0, 2.5, 0.1)
    engagement = st.slider("Engagement rating (1–10)", 1, 10, 6)
with c2:
    satisfaction = st.slider("Satisfaction rating (1–10)", 1, 10, 7)
    support_q = st.slider("Support tickets logged", 0, 10, 1)

c1, c2 = st.columns(2)
with c1:
    device = st.selectbox("Device used most", CAT_INFO["device_used_most_often"])
with c2:
    genre = st.selectbox("Favourite genre", CAT_INFO["genre_preference"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Churn", use_container_width=True):
    st.session_state["churn_inputs"] = dict(
        age=age, income=income, region=region,
        plan=plan, sub_len=sub_len, payment=payment,
        promo=promo, profiles=profiles, watch_time=watch_time,
        engagement=engagement, satisfaction=satisfaction, support_q=support_q,
        subscription_plan=plan, device_used_most_often=device, genre_preference=genre,
    )
    st.switch_page("pages/1_Prediction_Result.py")

st.markdown(
    """
    <div style="text-align:center;color:#444;font-size:.76rem;
    margin-top:36px;border-top:1px solid #222;padding-top:14px;">
    Netflix Customer Churn Prediction — Random Forest, trained on startup
    </div>
    """,
    unsafe_allow_html=True,
)
