import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn_core import (
    train_model, base_css, assign_tier, build_feature_row, INTERVENTIONS
)

st.set_page_config(
    page_title="Churn Result — Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(base_css(), unsafe_allow_html=True)

if "churn_inputs" not in st.session_state:
    st.warning("No customer details submitted yet — head back and fill in the form first.")
    if st.button("Go to customer details"):
        st.switch_page("app.py")
    st.stop()

inputs = st.session_state["churn_inputs"]
model, scaler, FEAT_NAMES, META = train_model()

feat_df, risk_sc = build_feature_row(inputs, FEAT_NAMES)
scaled = scaler.transform(feat_df)
prob = float(model.predict_proba(scaled)[0][1])
pred = int(model.predict(scaled)[0])
tier, tier_colour, res_cls = assign_tier(prob)
int_title, int_text = INTERVENTIONS[tier]

label = "Likely to churn" if pred == 1 else ("Borderline — worth watching" if tier == "Medium Risk" else "Likely to stay")

st.markdown(
    """
    <div class="hero">
        <p class="hero-title">Prediction result</p>
        <p class="hero-sub">Here's what the model thinks based on the profile you entered.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        f"""
        <div class="res-card {res_cls}">
            <div class="res-title">{label}</div>
            <div class="res-prob">Churn probability: <strong>{prob:.1%}</strong></div>
            <div class="res-tier">Risk tier: {tier}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob:.3f}")
    m2.metric("Risk score", f"{risk_sc:.4f}")
    m3.metric("Prediction", "Churn" if pred else "Retain")

    st.markdown(
        f"""
        <div class="intv">
            <div class="intv-title">Suggested next step: {int_title}</div>
            <div class="intv-text">{int_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sec-hdr">How the tiers break down</p>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Tier": ["High risk", "Medium risk", "Low risk"],
            "Probability": ["70% and up", "40%–69%", "Under 40%"],
            "What we'd do": [
                "Personal outreach plus a discount",
                "Automated email plus a small loyalty perk",
                "Nothing — standard experience",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.markdown('<p class="sec-hdr">Churn probability</p>', unsafe_allow_html=True)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": "#fff", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666", "tickfont": {"color": "#aaa"}},
            "bar": {"color": tier_colour, "thickness": 0.3},
            "bgcolor": "#1a1a1a", "bordercolor": "#333",
            "steps": [
                {"range": [0, 40], "color": "#0a2018"},
                {"range": [40, 70], "color": "#221800"},
                {"range": [70, 100], "color": "#1f0000"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.85, "value": prob * 100},
        },
    ))
    fig_g.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#141414",
        height=280, margin=dict(l=30, r=30, t=30, b=10),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<p class="sec-hdr">What drove this, roughly</p>', unsafe_allow_html=True)
    snap = {
        "Satisfaction": inputs["satisfaction"] / 10,
        "Engagement": inputs["engagement"] / 10,
        "Risk score": risk_sc,
        "Watch time": inputs["watch_time"] / 5,
        "Support tickets": min(inputs["support_q"] / 10, 1),
        "Sub. length": min(inputs["sub_len"] / 24, 1),
    }
    fig_b = go.Figure(go.Bar(
        x=list(snap.values()), y=list(snap.keys()), orientation="h",
        marker_color=["#27AE60", "#4A90D9", "#E50914", "#F5A623", "#9B59B6", "#7F8C8D"],
        text=[f"{v:.2f}" for v in snap.values()],
        textposition="outside", textfont={"color": "#fff"},
    ))
    fig_b.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#1a1a1a", font_color="#fff",
        height=230, xaxis=dict(range=[0, 1.35], gridcolor="#333"),
        yaxis=dict(gridcolor="#333"), margin=dict(l=10, r=40, t=8, b=8),
    )
    st.plotly_chart(fig_b, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model performance", "Feature importance", "About this model"])

with tab1:
    st.markdown(
        "We compared a few approaches before settling on the tuned Random "
        "Forest below; it had the best balance of recall (catching churners) "
        "and overall accuracy without overfitting to the majority class."
    )
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    perf_row = [
        META["accuracy"], META["precision"], META["recall"], META["f1"], META["auc"]
    ]
    fig_cmp = go.Figure(go.Bar(
        x=metrics_list, y=perf_row,
        marker_color="#E50914",
        text=[f"{v:.3f}" for v in perf_row],
        textposition="outside", textfont={"color": "#fff"},
    ))
    fig_cmp.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#1a1a1a", font_color="#fff",
        height=380, yaxis=dict(range=[0, 1.1], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        title=dict(text="Tuned Random Forest — test set metrics", font={"color": "#fff", "size": 14}),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

with tab2:
    st.markdown(
        "Which inputs the model leaned on most heavily when separating "
        "churners from loyal subscribers, across the training data."
    )
    fi = (
        pd.DataFrame({"Feature": FEAT_NAMES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True).tail(20)
    )
    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker_color="#E50914",
        text=[f"{v:.4f}" for v in fi["Importance"]],
        textposition="outside", textfont={"color": "#fff", "size": 9},
    ))
    fig_fi.update_layout(
        paper_bgcolor="#141414", plot_bgcolor="#1a1a1a", font_color="#fff",
        height=520, xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=10, r=80, t=20, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with tab3:
    st.markdown(
        f"""
        This app trains a Random Forest classifier on **{META['total_customers']:,}**
        Netflix subscriber records ({META['churn_rate']*100:.1f}% of whom churned)
        every time it starts up, rather than loading a saved model file. That
        keeps it from breaking when the scikit-learn version on the server
        doesn't match whatever was used to originally pickle the model — a
        recurring headache with shipping `.pkl` files.

        A quick rundown of the pipeline:

        - Categorical fields (plan, device, genre, region) are one-hot encoded.
        - The training set is rebalanced with SMOTE, since churners are the
          minority class.
        - Features are standardized before being fed to the forest.
        - 100 trees, Gini split criterion, `sqrt` max features, 70/30
          stratified train/test split.

        Worth keeping in mind: this is trained on a fairly small, synthetic
        dataset, so treat the probabilities as directional rather than exact.
        """
    )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Try another customer"):
    del st.session_state["churn_inputs"]
    st.switch_page("app.py")

st.markdown(
    """
    <div style="text-align:center;color:#444;font-size:.76rem;
    margin-top:30px;border-top:1px solid #222;padding-top:14px;">
    Netflix Customer Churn Prediction — Random Forest, trained on startup
    </div>
    """,
    unsafe_allow_html=True,
)
