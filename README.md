# 🎬 Netflix Customer Churn Predictor

A Netflix-themed Streamlit web app that loads a trained **Random Forest** model
to predict customer churn probability in real time.

---

## 📁 Repository Structure

```
netflix-churn-app/
├── app.py                          ← Streamlit application
├── requirements.txt                ← Python dependencies
├── netflix_churn_rf_model.pkl      ← Trained Random Forest model
├── netflix_churn_scaler.pkl        ← Fitted StandardScaler
├── netflix_feature_names.pkl       ← Ordered feature column names
├── netflix_app_meta.json           ← Dataset stats & model metrics
└── README.md                       ← This file
```

---

## 🚀 Local Setup

```bash
# 1. Clone / unzip the project
cd netflix-churn-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push to GitHub** — create a new repo and push this entire folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit — Netflix Churn Predictor"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your GitHub repo and branch (`main`)
   - Set **Main file path** → `app.py`
   - Click **Deploy**

> ⚠️ The `.pkl` files are ~3 MB total — well within GitHub's 100 MB limit.

---

## 🎨 App Features

| Feature | Details |
|---------|---------|
| **Live prediction** | Churn probability from 15 sidebar inputs |
| **Risk tiers** | 🔴 High / 🟡 Medium / 🟢 Low with colour-coded cards |
| **Plotly gauge** | Visual churn probability meter |
| **Intervention guide** | Automated retention strategy recommendation |
| **Model comparison** | Grouped bar chart (4 models) |
| **Feature importance** | Horizontal bar chart from trained RF |
| **Netflix dark theme** | `#141414` background · `#E50914` red accents |

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Decision Tree | 45.67% | 0.4691 | 0.4280 |
| Logistic Regression | 52.00% | 0.5714 | 0.5012 |
| Random Forest (Default) | 50.33% | 0.5552 | 0.5109 |
| **Random Forest (Tuned) ★** | **49.33%** | **0.5449** | **0.5337** |

---

## 🛠️ How the Model Was Built

1. **Dataset:** `netflix_large_user_data.xlsx` — 1,000 subscribers, 16 features
2. **Cleaning:** snake_case columns, duplicate removal, IQR outlier capping
3. **Feature engineering:**
   - `payment_delayed` (binary)
   - `plan_encoded` (ordinal: Basic=1, Standard=2, Premium=3)
   - `risk_score` (composite: satisfaction × 35% + engagement × 30% + payment × 20% + tenure × 15%)
   - One-hot encoding for plan, device, genre, region
4. **Preprocessing:** SMOTE oversampling → StandardScaler
5. **Model:** RandomForestClassifier (100 trees, Gini, sqrt features)
6. **Split:** 70/30 stratified

---

*JKUAT Data Science & Analytics — SCT213-C002-0094/2022*
