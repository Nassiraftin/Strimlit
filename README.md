# 🎬 Netflix Customer Churn Predictor

A Netflix-themed Streamlit web app with a **Random Forest** model that trains
automatically at startup — fully version-safe, no pickle files needed.

## 📁 Files

```
netflix-churn-app/
├── app.py                         ← Streamlit app (trains model at startup)
├── requirements.txt               ← Python dependencies
├── netflix_large_user_data.xlsx   ← Dataset (87 KB)
└── README.md
```

## 🚀 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

```bash
git init
git add .
git commit -m "Netflix Churn Predictor"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Then go to **share.streamlit.io** → New app → select repo → `app.py` → Deploy.

## ✅ Why no .pkl files?

The model trains at startup using `@st.cache_resource` (runs once, cached).
This avoids sklearn version mismatch errors between local and cloud environments.
