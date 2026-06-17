"""
Shared logic for the Netflix churn app: model training, feature
encoding, and the styling used across both pages. Keeping this in
one place means the input page and the results page can't drift
out of sync with each other.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

PLAN_ORDER = {"Basic": 1, "Standard": 2, "Premium": 3}

INTERVENTIONS = {
    "High Risk": (
        "Priority outreach",
        "This customer is worth a direct call or email from the retention "
        "team, paired with a discount offer. Waiting for them to cancel on "
        "their own is the expensive option.",
    ),
    "Medium Risk": (
        "Targeted nudge",
        "An automated email with a few recommended titles and a small "
        "loyalty perk (a free month, say) is usually enough to keep this "
        "group around a while longer.",
    ),
    "Low Risk": (
        "No action needed",
        "Nothing unusual here. Regular recommendations and the normal "
        "billing cycle are fine.",
    ),
}


def base_css() -> str:
    """CSS shared by both pages (Netflix-ish dark theme)."""
    return """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background:#141414 !important; color:#fff; }
    [data-testid="stHeader"] { background:#000 !important; }
    .block-container { padding-top:1.5rem !important; max-width:1100px; }
    .hero {
        background:linear-gradient(135deg,#E50914 0%,#831010 100%);
        border-radius:14px; padding:26px 34px; margin-bottom:22px;
        box-shadow:0 8px 40px rgba(229,9,20,.35);
    }
    .hero-title { font-size:1.8rem; font-weight:800; color:#fff; margin:0 0 4px; }
    .hero-sub { font-size:.92rem; color:rgba(255,255,255,.82); margin:0; }
    .sec-hdr {
        font-size:.8rem; font-weight:700; color:#E50914;
        border-left:4px solid #E50914; padding-left:10px;
        text-transform:uppercase; letter-spacing:.8px; margin:22px 0 12px;
    }
    .res-card { border-radius:14px; padding:30px 24px; text-align:center; margin-bottom:18px; }
    .res-churn  { background:linear-gradient(135deg,#E50914,#7a0000); box-shadow:0 6px 32px rgba(229,9,20,.45); }
    .res-medium { background:linear-gradient(135deg,#F5A623,#a86e00); box-shadow:0 6px 32px rgba(245,166,35,.35); }
    .res-safe   { background:linear-gradient(135deg,#27AE60,#145c30); box-shadow:0 6px 32px rgba(39,174,96,.35); }
    .res-title { font-size:1.5rem; font-weight:800; color:#fff; margin-bottom:6px; }
    .res-prob  { font-size:1.02rem; color:rgba(255,255,255,.88); }
    .res-tier  { font-size:.88rem; color:rgba(255,255,255,.72); margin-top:6px; font-weight:600; }
    .intv {
        background:#1a1a1a; border:1px solid #333; border-left:5px solid #E50914;
        border-radius:8px; padding:16px 18px; margin-top:12px;
    }
    .intv-title { font-weight:700; color:#E50914; font-size:.86rem; margin-bottom:6px; }
    .intv-text  { color:#ccc; font-size:.86rem; line-height:1.65; }
    .stButton > button {
        background:linear-gradient(135deg,#E50914,#a00000) !important;
        color:#fff !important; border:none !important;
        font-size:.98rem !important; font-weight:700 !important;
        letter-spacing:.4px !important; border-radius:8px !important;
        padding:12px !important;
        box-shadow:0 4px 18px rgba(229,9,20,.4) !important;
    }
    .stButton > button:hover { transform:translateY(-1px) !important; }
    .stTabs [data-baseweb="tab-list"] { background:#000; border-bottom:1px solid #2a2a2a; }
    .stTabs [data-baseweb="tab"] { color:#888 !important; }
    .stTabs [aria-selected="true"] { color:#E50914 !important; border-bottom:2px solid #E50914 !important; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """


@st.cache_resource(show_spinner="Training the churn model on the Netflix dataset…")
def train_model():
    base_dir = os.path.dirname(__file__)
    df = pd.read_excel(os.path.join(base_dir, "netflix_large_user_data.xlsx"))

    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[\s/()]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
        .str.strip("_")
    )

    pay_col = "payment_history_ontime_delayed"
    income_col = "monthly_income"

    df["churn"] = (df["churn_status_yes_no"].str.strip().str.lower() == "yes").astype(int)
    df["payment_delayed"] = (df[pay_col].str.strip().str.lower() == "delayed").astype(int)
    df["plan_encoded"] = df["subscription_plan"].map(PLAN_ORDER)

    sat_n = 1 - (df["customer_satisfaction_score_110"] - 1) / 9
    eng_n = 1 - (df["engagement_rate_110"] - 1) / 9
    sub_r = 1 - (df["subscription_length_months"] / 24)
    df["risk_score"] = (
        sat_n * 0.35 + eng_n * 0.30 + df["payment_delayed"] * 0.20 + sub_r.clip(0, 1) * 0.15
    ).round(4)

    feature_cols = [
        "subscription_length_months", "customer_satisfaction_score_110",
        "daily_watch_time_hours", "engagement_rate_110",
        "support_queries_logged", "age",
        income_col, "promotional_offers_used",
        "number_of_profiles_created", "payment_delayed",
        "plan_encoded", "risk_score",
        "subscription_plan", "device_used_most_often",
        "genre_preference", "region",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "churn"

    model_df = df[feature_cols + [target]].copy().dropna(subset=[target])
    cat_cols = model_df.select_dtypes(include=["object", "string"]).columns.tolist()
    model_enc = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
    bool_cols = model_enc.select_dtypes(bool).columns
    model_enc[bool_cols] = model_enc[bool_cols].astype(int)

    X = model_enc.drop(columns=[target])
    y = model_enc[target].astype(int)
    feat_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_res)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        max_features="sqrt", criterion="gini", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_res)

    yp = rf.predict(X_test_sc)
    yprob = rf.predict_proba(X_test_sc)[:, 1]

    cat_info = {}
    for col in [pay_col, "subscription_plan", "device_used_most_often", "genre_preference", "region"]:
        if col in df.columns:
            cat_info[col] = sorted(df[col].dropna().unique().tolist())

    meta = {
        "accuracy": float(accuracy_score(y_test, yp)),
        "precision": float(precision_score(y_test, yp)),
        "recall": float(recall_score(y_test, yp)),
        "f1": float(f1_score(y_test, yp)),
        "auc": float(roc_auc_score(y_test, yprob)),
        "total_customers": len(df),
        "churn_rate": float(df["churn"].mean()),
        "age_min": int(df["age"].min()),
        "age_max": int(df["age"].max()),
        "income_min": int(df[income_col].min()),
        "income_max": int(df[income_col].max()),
        "income_mean": float(df[income_col].mean()),
        "cat_info": cat_info,
        "pay_col": pay_col,
    }
    return rf, scaler, feat_names, meta


def assign_tier(prob: float):
    if prob >= 0.70:
        return "High Risk", "#E50914", "res-churn"
    if prob >= 0.40:
        return "Medium Risk", "#F5A623", "res-medium"
    return "Low Risk", "#27AE60", "res-safe"


def build_feature_row(inputs: dict, feat_names: list):
    pay_delayed = 1 if inputs["payment"] == "Delayed" else 0
    plan_enc = PLAN_ORDER.get(inputs["plan"], 1)

    sat_n = 1 - (inputs["satisfaction"] - 1) / 9
    eng_n = 1 - (inputs["engagement"] - 1) / 9
    sub_risk = max(0, min(1, 1 - inputs["sub_len"] / 24))
    risk_score = round(sat_n * 0.35 + eng_n * 0.30 + pay_delayed * 0.20 + sub_risk * 0.15, 4)

    row = {
        "subscription_length_months": inputs["sub_len"],
        "customer_satisfaction_score_110": inputs["satisfaction"],
        "daily_watch_time_hours": inputs["watch_time"],
        "engagement_rate_110": inputs["engagement"],
        "support_queries_logged": inputs["support_q"],
        "age": inputs["age"],
        "monthly_income": inputs["income"],
        "promotional_offers_used": inputs["promo"],
        "number_of_profiles_created": inputs["profiles"],
        "payment_delayed": pay_delayed,
        "plan_encoded": plan_enc,
        "risk_score": risk_score,
    }

    dummies = {
        "subscription_plan": ["Premium", "Standard"],
        "device_used_most_often": ["Laptop", "Mobile", "Smart TV", "Tablet"],
        "genre_preference": ["Comedy", "Documentary", "Drama", "Romance", "Sci-Fi", "Thriller"],
        "region": ["Asia", "Europe", "North America", "South America"],
    }
    for col, cats in dummies.items():
        for cat in cats:
            row[f"{col}_{cat}"] = 1 if inputs.get(col) == cat else 0

    df_row = pd.DataFrame([row])
    for c in feat_names:
        if c not in df_row.columns:
            df_row[c] = 0
    return df_row[feat_names], risk_score
