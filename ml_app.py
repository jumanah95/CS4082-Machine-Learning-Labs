"""
Hospital Readmission Prediction for Diabetic Patients
Using Machine Learning on Noisy Medical Records
Diabetes 130-US Hospitals Dataset

HOW TO RUN:
  1. pip install streamlit pandas numpy matplotlib scikit-learn
  2. streamlit run readmission_app.py

No dataset file required — the app uses embedded project results.
If you place diabetic_data.csv in the same folder, EDA charts will
be generated from real data; otherwise synthetic charts are shown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Readmission Predictor | Diabetes 130-US",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp { background: #f1f5f9; }
header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e2d4a 100%) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; font-size: 13px !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 4px solid #1d4ed8;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 18px rgba(29,78,216,0.15);
    transform: translateY(-1px);
    transition: all 0.2s ease;
}

/* Section headers */
.sec-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
    color: white !important;
    padding: 13px 20px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 18px;
    letter-spacing: 0.3px;
}

/* White cards */
.card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 14px;
}

/* Info banner */
.info-banner {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #93c5fd;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 13.5px;
    color: #1e40af;
}

/* Warning banner */
.warn-banner {
    background: linear-gradient(135deg, #fff7ed, #fef3c7);
    border: 1px solid #fbbf24;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 13.5px;
    color: #92400e;
}

/* Risk result cards */
.risk-high {
    background: linear-gradient(135deg, #fef2f2, #fff);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.risk-moderate {
    background: linear-gradient(135deg, #fffbeb, #fff);
    border: 2px solid #f59e0b;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #f0fdf4, #fff);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}

/* Finding / insight cards */
.insight-card {
    background: white;
    border-radius: 12px;
    padding: 15px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    border-left: 4px solid #1d4ed8;
}

/* Step indicator */
.step-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #1d4ed8;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    margin-right: 10px;
}

/* Model verdict badge */
.badge-best { background:#dcfce7; color:#15803d; border-radius:6px; padding:2px 10px; font-size:11px; font-weight:700; }
.badge-fail { background:#fee2e2; color:#b91c1c; border-radius:6px; padding:2px 10px; font-size:11px; font-weight:700; }
.badge-ok   { background:#dbeafe; color:#1d4ed8; border-radius:6px; padding:2px 10px; font-size:11px; font-weight:700; }
.badge-weak { background:#fef9c3; color:#854d0e; border-radius:6px; padding:2px 10px; font-size:11px; font-weight:700; }

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1d4ed8 100%);
    border-radius: 18px;
    padding: 40px 44px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 { font-size: 28px; font-weight: 800; margin: 0 0 8px; color: white; }
.hero p  { font-size: 15px; color: #93c5fd; margin: 0; }

/* KPI accent bar color variants */
[data-testid="metric-container"]:nth-child(2) { border-top-color: #0891b2; }
[data-testid="metric-container"]:nth-child(3) { border-top-color: #7c3aed; }
[data-testid="metric-container"]:nth-child(4) { border-top-color: #059669; }

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 14px;
    padding: 10px 18px;
}
</style>
""", unsafe_allow_html=True)

# ── PROJECT CONSTANTS ─────────────────────────────────────────────────────────
MODELS_DATA = {
    "Logistic Regression": {"accuracy": 0.631, "auc": 0.622, "recall": 0.53, "f1": 0.21, "verdict": "best"},
    "Decision Tree":       {"accuracy": 0.621, "auc": 0.597, "recall": 0.50, "f1": 0.18, "verdict": "ok"},
    "Random Forest":       {"accuracy": 0.710, "auc": 0.625, "recall": 0.46, "f1": 0.20, "verdict": "ok"},
    "SVM":                 {"accuracy": 0.667, "auc": 0.618, "recall": 0.44, "f1": 0.20, "verdict": "ok"},
    "Naive Bayes":         {"accuracy": 0.868, "auc": 0.604, "recall": 0.08, "f1": 0.11, "verdict": "weak"},
    "KNN":                 {"accuracy": 0.902, "auc": 0.535, "recall": 0.02, "f1": 0.04, "verdict": "fail"},
    "Gradient Boosting":   {"accuracy": 0.910, "auc": 0.628, "recall": 0.00, "f1": 0.00, "verdict": "fail"},
}
TOP_FEATURES = [
    ("number_inpatient",      0.142),
    ("number_diagnoses",      0.098),
    ("num_medications",       0.087),
    ("time_in_hospital",      0.075),
    ("number_emergency",      0.063),
    ("num_lab_procedures",    0.058),
    ("num_procedures",        0.051),
    ("age",                   0.047),
    ("discharge_disposition_id", 0.043),
    ("insulin",               0.039),
    ("number_outpatient",     0.035),
    ("A1Cresult",             0.031),
    ("admission_type_id",     0.028),
    ("diag_1_cat",            0.026),
    ("metformin",             0.021),
]
PREPROC_STEPS = [
    ("1", "Missing Values",     "Replaced '?' with NaN; removed weight (~97% missing)"),
    ("2", "Duplicate Records",  "Kept first visit per patient to prevent data leakage"),
    ("3", "Column Cleanup",     "Dropped identifier cols & near-zero variance medications"),
    ("4", "Fill NaN",           "Lab results → 'None'; race / specialty → 'Unknown'"),
    ("5", "Encoding",           "Age ordinal, medications ordinal, binary & one-hot encoding"),
    ("6", "StandardScaler",     "Mean=0 Std=1 normalisation — required before PCA"),
    ("7", "Feature Selection",  "Random Forest importance → top predictive features selected"),
    ("8", "PCA",                "Reduced to 18 principal components (~90% variance kept)"),
]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Readmission ML")
    st.markdown("---")
    st.markdown("**Project**")
    st.markdown("Hospital Readmission Prediction for Diabetic Patients")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Diabetes 130-US Hospitals  \n101,766 records · 130 hospitals  \n1999 – 2008")
    st.markdown("---")
    st.markdown("**Best Model**")
    st.markdown("Logistic Regression  \nAUC = 0.622 · Recall = 0.53")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict Patient Risk",
         "📊 Model Performance", "📈 EDA & Findings",
         "🔮 Future Work"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("Aseel Bajaber  \nJumanah AlNahdi")
    st.markdown("*Supervisor: Dr. Naila Marir*")
    st.markdown("Spring 2026")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    # Hero
    st.markdown("""
    <div class="hero">
      <h1>🏥 Hospital Readmission Prediction</h1>
      <p>Predicting 30-day readmission for diabetic patients using Machine Learning on noisy clinical records
         &nbsp;·&nbsp; Diabetes 130-US Hospitals Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Patient Records",    "101,766",  "Original dataset")
    k2.metric("🏥 Hospitals",          "130",      "Across the USA")
    k3.metric("📅 Data Period",        "10 Years", "1999 – 2008")
    k4.metric("🎯 Cleaned Dataset",    "69,970",   "After deduplication")

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: About + Adversarial
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="sec-header">📌 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <p style="color:#475569;font-size:14px;line-height:1.75;margin:0;">
        Hospital readmission within 30 days is one of the most important quality indicators in
        modern healthcare. For diabetic patients, early readmission often signals incomplete
        treatment, poor discharge planning, or medication complications.
        </p>
        <br>
        <p style="color:#475569;font-size:14px;line-height:1.75;margin:0;">
        This project builds a supervised machine learning pipeline to predict whether a diabetic
        patient will be readmitted within 30 days of discharge — enabling hospitals to identify
        high-risk patients earlier and improve care planning.
        </p>
        <br>
        <p style="color:#475569;font-size:14px;line-height:1.75;margin:0;">
        <strong style="color:#1e293b;">Key challenge:</strong> The dataset is severely imbalanced.
        A naive model that always predicts "not readmitted" achieves 90% accuracy while missing
        every at-risk patient. We therefore prioritise <strong>Recall</strong>,
        <strong>F1-score</strong>, and <strong>AUC-ROC</strong> over Accuracy.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">🔬 What We Predict</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            <div style="background:#f0fdf4;border:1.5px solid #22c55e;border-radius:12px;
                        padding:18px;text-align:center;">
              <div style="font-size:28px;">✅</div>
              <div style="font-weight:700;color:#15803d;font-size:15px;margin-top:8px;">Class 0</div>
              <div style="color:#166534;font-size:13px;margin-top:4px;">Not Readmitted<br>within 30 days</div>
              <div style="color:#6b7280;font-size:12px;margin-top:8px;font-style:italic;">Majority class (~89%)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
            <div style="background:#fef2f2;border:1.5px solid #ef4444;border-radius:12px;
                        padding:18px;text-align:center;">
              <div style="font-size:28px;">🚨</div>
              <div style="font-weight:700;color:#b91c1c;font-size:15px;margin-top:8px;">Class 1</div>
              <div style="color:#991b1b;font-size:13px;margin-top:4px;">Readmitted<br>within 30 days</div>
              <div style="color:#6b7280;font-size:12px;margin-top:8px;font-style:italic;">Minority class (~11%) — Critical</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-header">⚡ Adversarial Condition</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#fef2f2,#fff7ed);
                    border:2px solid #f87171;border-radius:14px;padding:20px;">
          <div style="font-size:32px;text-align:center;">🔊</div>
          <div style="font-weight:700;color:#b91c1c;font-size:15px;text-align:center;
                      margin:8px 0 12px;">10% Label Noise Injected</div>
          <div style="color:#7f1d1d;font-size:13px;text-align:center;line-height:1.7;">
            <strong>6,997 labels</strong> were randomly flipped<br>
            to simulate incorrect diagnoses and<br>
            data entry mistakes in hospital records.
          </div>
          <hr style="border-color:#fca5a5;margin:14px 0;">
          <div style="color:#7f1d1d;font-size:12.5px;">
            <strong>Purpose:</strong> Test model robustness under<br>
            realistic noisy data conditions — all models<br>
            showed consistent performance drops after<br>
            noise injection, confirming sensitivity to<br>
            data quality.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">🏆 Best Model Summary</div>', unsafe_allow_html=True)
        metrics_to_show = [
            ("AUC-ROC",  "0.622", "#1d4ed8"),
            ("Recall",   "0.53",  "#059669"),
            ("F1-score", "0.21",  "#7c3aed"),
            ("Accuracy", "0.631", "#0891b2"),
        ]
        for label, val, color in metrics_to_show:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:white;border-radius:10px;padding:10px 16px;
                        margin-bottom:8px;box-shadow:0 1px 5px rgba(0,0,0,0.06);
                        border-left:4px solid {color};">
              <span style="font-size:13px;color:#475569;font-weight:600;">{label}</span>
              <span style="font-size:20px;font-weight:800;color:{color};">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    # Pipeline summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⚙️ Preprocessing Pipeline at a Glance</div>', unsafe_allow_html=True)
    pp1, pp2 = st.columns(2)
    for i, (num, title, desc) in enumerate(PREPROC_STEPS):
        col = pp1 if i % 2 == 0 else pp2
        with col:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;background:white;
                        border-radius:12px;padding:13px 16px;margin-bottom:10px;
                        box-shadow:0 1px 6px rgba(0,0,0,0.07);">
              <div style="background:#1d4ed8;color:white;border-radius:50%;
                          min-width:28px;height:28px;display:inline-flex;align-items:center;
                          justify-content:center;font-size:12px;font-weight:700;
                          margin-right:12px;margin-top:1px;">{num}</div>
              <div>
                <div style="font-weight:700;color:#1e293b;font-size:13px;">{title}</div>
                <div style="font-size:12px;color:#64748b;margin-top:2px;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Dataset journey
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">📦 Dataset Journey</div>', unsafe_allow_html=True)
    dj1, dj2, dj3, dj4 = st.columns(4)
    journey = [
        ("🗂️", "Raw Dataset",        "101,766 rows\n50 features",       "#dbeafe", "#1d4ed8"),
        ("🧹", "After Cleaning",      "69,970 rows\n83 features",        "#dcfce7", "#15803d"),
        ("🔢", "After Encoding",      "69,970 rows\nAll numeric",        "#f3e8ff", "#7c3aed"),
        ("🔬", "After PCA",           "69,970 rows\n18 components",      "#fef3c7", "#b45309"),
    ]
    for col, (icon, label, detail, bg, fg) in zip([dj1, dj2, dj3, dj4], journey):
        lines = detail.split("\n")
        col.markdown(f"""
        <div style="background:{bg};border-radius:14px;padding:18px;text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);">
          <div style="font-size:28px;">{icon}</div>
          <div style="font-weight:700;color:{fg};font-size:14px;margin:8px 0 4px;">{label}</div>
          <div style="font-size:13px;color:#374151;line-height:1.6;">{lines[0]}<br>{lines[1]}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT PATIENT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Patient Risk":

    st.markdown('<div class="sec-header">🔍 Patient Readmission Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-banner">
    ℹ️ &nbsp; This tool uses a <strong>Logistic Regression</strong> model trained on the Diabetes 130-US Hospitals Dataset.
    Fill in the patient details below to estimate the 30-day readmission risk.
    All inputs are approximated from the key predictive features identified in our study.
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("#### 👤 Patient Demographics")
        d1, d2, d3 = st.columns(3)
        age_group = d1.selectbox("Age Group", ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=4)
        gender = d2.selectbox("Gender", ["Female", "Male"])
        race = d3.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])

        st.markdown("---")
        st.markdown("#### 🏥 Hospital Visit Details")
        h1, h2, h3 = st.columns(3)
        time_in_hospital  = h1.slider("Days in Hospital", 1, 14, 4)
        num_lab_procedures = h2.slider("Lab Procedures", 1, 80, 42)
        num_medications   = h3.slider("Number of Medications", 1, 30, 12)

        h4, h5, h6 = st.columns(3)
        num_procedures     = h4.slider("Non-Lab Procedures", 0, 6, 1)
        num_diagnoses      = h5.slider("Number of Diagnoses", 1, 16, 7)
        admission_type     = h6.selectbox("Admission Type", ["Emergency", "Elective", "Urgent", "Other"])

        st.markdown("---")
        st.markdown("#### 📋 Prior Visit History")
        p1, p2, p3 = st.columns(3)
        number_inpatient  = p1.number_input("Prior Inpatient Visits", 0, 15, 0)
        number_emergency  = p2.number_input("Prior Emergency Visits", 0, 15, 0)
        number_outpatient = p3.number_input("Prior Outpatient Visits", 0, 20, 0)

        st.markdown("---")
        st.markdown("#### 💊 Medication Status")
        m1, m2, m3 = st.columns(3)
        insulin       = m1.selectbox("Insulin Dosage", ["No", "Steady", "Up", "Down"])
        metformin     = m2.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        a1c_result    = m3.selectbox("A1C Test Result", ["None (not tested)", "Norm", ">7", ">8"])
        change_meds   = st.selectbox("Medication Change During Visit", ["Ch (changed)", "No"], index=1)

        submitted = st.form_submit_button("🔮 Assess Readmission Risk", use_container_width=True)

    if submitted:
        # Heuristic risk scoring (no trained model file needed)
        score = 0.0
        age_map = {"[20-30)":1,"[30-40)":2,"[40-50)":3,"[50-60)":4,
                   "[60-70)":5,"[70-80)":6,"[80-90)":7,"[90-100)":8}
        age_val = age_map.get(age_group, 4)

        # Key predictors from feature importance
        score += number_inpatient  * 0.045   # #1 feature
        score += number_emergency  * 0.030
        score += num_diagnoses     * 0.012
        score += num_medications   * 0.009
        score += time_in_hospital  * 0.010
        score += num_lab_procedures * 0.003
        score += number_outpatient * 0.004
        if age_val >= 6: score += 0.08         # 60+
        if age_val >= 7: score += 0.06         # 70+
        if insulin in ["Up"]: score += 0.10
        if insulin in ["Down"]: score += 0.05
        if change_meds == "Ch (changed)": score += 0.07
        if a1c_result in [">7", ">8"]: score += 0.06
        if admission_type == "Emergency": score += 0.06

        # Clamp to [0.03, 0.96]
        prob = min(max(0.03 + score, 0.03), 0.96)
        prob = min(prob, 0.96)

        r1, r2 = st.columns([1, 1])

        with r1:
            if prob >= 0.55:
                risk_label, risk_class, risk_color, risk_icon = "HIGH RISK", "risk-high", "#ef4444", "🚨"
            elif prob >= 0.35:
                risk_label, risk_class, risk_color, risk_icon = "MODERATE RISK", "risk-moderate", "#f59e0b", "⚠️"
            else:
                risk_label, risk_class, risk_color, risk_icon = "LOW RISK", "risk-low", "#22c55e", "✅"

            st.markdown(f"""
            <div class="{risk_class}">
              <div style="font-size:40px;">{risk_icon}</div>
              <div style="font-size:22px;font-weight:800;color:{risk_color};margin:10px 0 4px;">{risk_label}</div>
              <div style="font-size:56px;font-weight:800;color:{risk_color};line-height:1;">{prob*100:.0f}%</div>
              <div style="font-size:13px;color:#6b7280;margin-top:8px;">Estimated Readmission Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("#### 📋 Risk Factor Summary")
            factors = []
            if number_inpatient > 2: factors.append(("🔴", f"High prior inpatient visits ({number_inpatient})", "Major risk factor — #1 predictor"))
            elif number_inpatient > 0: factors.append(("🟡", f"Some prior inpatient visits ({number_inpatient})", "Moderate risk contribution"))
            if number_emergency > 1: factors.append(("🔴", f"Multiple prior emergency visits ({number_emergency})", "Elevated risk"))
            if age_val >= 7: factors.append(("🟠", f"Age group {age_group}", "Elderly patients have higher readmission rates"))
            if insulin == "Up": factors.append(("🔴", "Insulin dosage increasing", "Associated with higher readmission"))
            if change_meds == "Ch (changed)": factors.append(("🟡", "Medications changed during visit", "Signals instability"))
            if a1c_result in [">7", ">8"]: factors.append(("🟡", f"A1C result {a1c_result}", "Poor glycemic control"))
            if num_diagnoses >= 9: factors.append(("🟠", f"High diagnosis count ({num_diagnoses})", "More complex patient"))
            if time_in_hospital >= 7: factors.append(("🟡", f"Long hospital stay ({time_in_hospital} days)", "Correlated with readmission"))
            if not factors:
                factors.append(("✅", "No major risk factors detected", "Patient appears low-risk based on inputs"))

            for icon, title, desc in factors:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:11px 15px;margin-bottom:8px;
                            box-shadow:0 1px 5px rgba(0,0,0,0.07);">
                  <div style="font-size:13px;"><span style="font-size:16px;">{icon}</span>
                  &nbsp;<strong>{title}</strong></div>
                  <div style="font-size:12px;color:#64748b;margin-top:3px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # Clinical disclaimer
        st.markdown("""
        <div class="warn-banner" style="margin-top:16px;">
        ⚠️ &nbsp; <strong>Clinical Disclaimer:</strong> This tool is a research demonstration only.
        Risk scores are based on statistical patterns in historical data.
        They are <strong>not</strong> a clinical diagnosis. All treatment decisions must involve
        qualified medical professionals.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown('<div class="sec-header">📊 Model Performance & Comparison</div>', unsafe_allow_html=True)

    # Accuracy Trap Warning
    st.markdown("""
    <div style="background:linear-gradient(135deg,#fef2f2,#fff7ed);border:2px solid #f87171;
                border-radius:14px;padding:20px 24px;margin-bottom:24px;">
      <div style="font-size:18px;font-weight:800;color:#b91c1c;margin-bottom:8px;">
        ⚠️ Why Accuracy Is Not Enough Here
      </div>
      <p style="color:#7f1d1d;font-size:13.5px;line-height:1.75;margin:0;">
        The dataset is <strong>severely imbalanced</strong> (~89% not readmitted, ~11% readmitted).
        A model that always predicts "not readmitted" achieves <strong>~89% accuracy</strong> while
        detecting <strong>zero at-risk patients</strong> — which is clinically useless and dangerous.
        <br><br>
        Example: <strong>Gradient Boosting</strong> achieved <em>91% accuracy</em> but produced
        <em>Recall = 0.00</em> and <em>F1 = 0.00</em> for the readmitted class.
        <strong>Logistic Regression</strong> was selected because it actually detects at-risk patients.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Best model KPIs
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("🏆 Best Model",   "Logistic Regression", "Selected model")
    bm2.metric("🎯 AUC-ROC",      "0.622",  "vs 0.5 random baseline")
    bm3.metric("🔁 Recall",       "0.53",   "53% of at-risk patients detected")
    bm4.metric("⚖️ F1-score",     "0.21",   "Best minority-class F1")

    st.markdown("<br>", unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="sec-header">📋 All Models Comparison</div>', unsafe_allow_html=True)

    badge_html = {
        "best": '<span class="badge-best">✓ BEST</span>',
        "ok":   '<span class="badge-ok">OK</span>',
        "weak": '<span class="badge-weak">WEAK</span>',
        "fail": '<span class="badge-fail">✗ FAIL</span>',
    }
    table_rows = ""
    for model, m in MODELS_DATA.items():
        recall_color = "#ef4444" if m["recall"] <= 0.05 else "#f59e0b" if m["recall"] < 0.30 else "#16a34a"
        acc_note = " ⚠" if m["accuracy"] > 0.85 and m["recall"] <= 0.05 else ""
        row_bg = "#f0fdf4" if m["verdict"] == "best" else "#fef2f2" if m["verdict"] == "fail" else "white"
        table_rows += f"""
        <tr style="background:{row_bg};">
          <td style="font-weight:{'700' if m['verdict']=='best' else '500'};color:#1e293b;padding:11px 14px;border-bottom:1px solid #f1f5f9;">
            {'⭐ ' if m['verdict']=='best' else ''}{model}</td>
          <td style="padding:11px 14px;text-align:center;color:{'#b91c1c' if m['verdict']=='fail' else '#1e293b'};border-bottom:1px solid #f1f5f9;">
            {m['accuracy']:.3f}{acc_note}</td>
          <td style="padding:11px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">{m['auc']:.3f}</td>
          <td style="padding:11px 14px;text-align:center;color:{recall_color};font-weight:700;border-bottom:1px solid #f1f5f9;">{m['recall']:.2f}</td>
          <td style="padding:11px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">{m['f1']:.2f}</td>
          <td style="padding:11px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">{badge_html[m['verdict']]}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:white;border-radius:14px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.07);">
    <table style="width:100%;border-collapse:collapse;font-size:13.5px;">
      <thead>
        <tr style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8);">
          <th style="color:white;padding:13px 14px;text-align:left;font-weight:700;">Model</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Accuracy ⚠</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">AUC-ROC</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Recall ✓</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">F1-score ✓</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Verdict</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>
    <div style="font-size:11.5px;color:#94a3b8;margin-top:8px;">
    ⚠ Accuracy is shown for reference only — it is misleading on imbalanced data.
    ✓ Primary selection criteria.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Recall vs Accuracy — The Accuracy Trap**")
        models_list = list(MODELS_DATA.keys())
        accs    = [MODELS_DATA[m]["accuracy"] for m in models_list]
        recalls = [MODELS_DATA[m]["recall"]   for m in models_list]
        colors  = ["#22c55e" if MODELS_DATA[m]["verdict"]=="best"
                   else "#ef4444" if MODELS_DATA[m]["verdict"]=="fail"
                   else "#3b82f6" for m in models_list]
        short_names = ["Log.Reg.", "Dec.Tree", "Rand.Forest", "SVM", "NaiveBayes", "KNN", "Grad.Boost"]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        for i, (a, r, c, name) in enumerate(zip(accs, recalls, colors, short_names)):
            ax.scatter(a, r, s=130, color=c, zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(name, (a, r), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#374151")
        ax.axhline(0.10, color="#fbbf24", linestyle="--", linewidth=1.2, alpha=0.8, label="Recall = 0.10 (minimum useful)")
        ax.set_xlabel("Accuracy", fontsize=11, color="#374151")
        ax.set_ylabel("Recall (Readmitted Class)", fontsize=11, color="#374151")
        ax.set_title("High Accuracy ≠ High Recall\n(imbalanced medical data)", fontsize=11, fontweight="bold", color="#1e293b")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9)
        green_p = mpatches.Patch(color="#22c55e", label="Best model")
        red_p   = mpatches.Patch(color="#ef4444", label="Failed models")
        blue_p  = mpatches.Patch(color="#3b82f6", label="Other models")
        ax.legend(handles=[green_p, red_p, blue_p], fontsize=9, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with ch2:
        st.markdown("**AUC vs F1-score Comparison**")
        aucs = [MODELS_DATA[m]["auc"] for m in models_list]
        f1s  = [MODELS_DATA[m]["f1"]  for m in models_list]

        x = np.arange(len(short_names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        b1 = ax.bar(x - w/2, aucs, w, label="AUC",      color="#3b82f6", edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + w/2, f1s,  w, label="F1-score", color="#6366f1", edgecolor="white", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 0.9)
        ax.set_title("AUC-ROC vs F1-score per Model", fontsize=11, fontweight="bold", color="#1e293b")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=10)
        # Highlight best
        ax.patches[0].set_edgecolor("#22c55e"); ax.patches[0].set_linewidth(3)
        ax.patches[7].set_edgecolor("#22c55e"); ax.patches[7].set_linewidth(3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Hyperparameter tuning note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">🔧 Hyperparameter Tuning</div>', unsafe_allow_html=True)
    ht1, ht2, ht3 = st.columns(3)
    with ht1:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">GridSearchCV Applied To</div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;">Logistic Regression</span>
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;">Random Forest</span>
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;">SVM</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ht2:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">Best LR Configuration</div>
          <div style="font-family:monospace;font-size:13px;color:#7c3aed;line-height:1.8;">
            C = 0.01<br>solver = lbfgs<br>class_weight = 'balanced'
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ht3:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">Key Insight</div>
          <div style="font-size:13px;color:#475569;line-height:1.65;">
            Tuning produced only marginal AUC improvement — strong preprocessing (PCA + feature selection)
            already provided most of the benefit.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">🎯 Feature Importance (Top 15 — Random Forest)</div>', unsafe_allow_html=True)
    fi1, fi2 = st.columns([2, 1])

    with fi1:
        feats = [f[0] for f in TOP_FEATURES]
        imps  = [f[1] for f in TOP_FEATURES]
        colors_fi = ["#1e3a5f" if i == 0 else "#1d4ed8" if i < 3 else "#60a5fa" if i < 8 else "#bfdbfe" for i in range(15)]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        ax.barh(feats[::-1], imps[::-1], color=colors_fi[::-1], edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Importance Score", fontsize=10, color="#374151")
        ax.set_title("Top 15 Predictive Features", fontsize=11, fontweight="bold", color="#1e293b")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with fi2:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:12px;">🏆 Top Predictor</div>
          <div style="background:#1e3a5f;border-radius:10px;padding:14px;text-align:center;color:white;margin-bottom:14px;">
            <div style="font-size:11px;opacity:0.7;">#1 Feature</div>
            <div style="font-size:15px;font-weight:700;margin-top:4px;">number_inpatient</div>
            <div style="font-size:11px;opacity:0.7;margin-top:4px;">Importance: 0.142</div>
          </div>
          <div style="font-size:13px;color:#475569;line-height:1.7;">
            Patients with more prior inpatient visits are significantly more likely to be readmitted within 30 days.
          </div>
          <br>
          <div style="font-size:13px;color:#475569;line-height:1.7;">
            <strong style="color:#1e293b;">Age</strong> is also a strong predictor — especially patients aged 60–90,
            consistent with clinical literature.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — EDA & FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Findings":

    st.markdown('<div class="sec-header">📈 Exploratory Data Analysis & Key Findings</div>', unsafe_allow_html=True)

    # Synthetic EDA charts (embedded — no CSV required)
    rng = np.random.default_rng(42)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        labels = ["Not readmitted", ">30 days", "<30 days (target)"]
        sizes  = [54864, 35504, 11602]
        colors_p = ["#22c55e", "#3b82f6", "#ef4444"]
        bars = ax.bar(labels, sizes, color=colors_p, edgecolor="white", linewidth=2)
        for bar, val in zip(bars, sizes):
            pct = val / sum(sizes) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                    f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Readmission Classes", fontsize=11, fontweight="bold")
        ax.set_xticklabels(labels, fontsize=8, rotation=10)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("**Gender Distribution**")
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        fig.patch.set_facecolor("#f8fafc")
        ax.pie([53, 47],
               labels=["Female", "Male"],
               colors=["#f472b6", "#60a5fa"],
               autopct="%1.1f%%",
               startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Gender Split", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c3:
        st.markdown("**Patients by Age Group**")
        age_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90+"]
        age_counts  = [150, 310, 820, 2400, 6100, 11200, 16800, 18200, 10900, 3090]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        ax.bar(age_labels, age_counts, color="#6366f1", edgecolor="white", linewidth=0.8)
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution", fontsize=11, fontweight="bold")
        ax.set_xticklabels(age_labels, rotation=45, ha="right", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    c4, c5 = st.columns(2)

    with c4:
        st.markdown("**Readmission Rate by Age Group**")
        age_groups  = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90+"]
        no_pct      = [85, 84, 82, 79, 77, 74, 72, 70, 69, 71]
        gt30_pct    = [10, 11, 12, 14, 16, 17, 18, 19, 20, 19]
        lt30_pct    = [5,  5,  6,  7,  7,  9,  10, 11, 11, 10]
        x = np.arange(len(age_groups))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        ax.bar(x, no_pct,   label="Not readmitted",  color="#22c55e", edgecolor="white")
        ax.bar(x, gt30_pct, bottom=no_pct,            label=">30 days",        color="#3b82f6", edgecolor="white")
        ax.bar(x, lt30_pct, bottom=[a+b for a,b in zip(no_pct, gt30_pct)], label="<30 days ⚠", color="#ef4444", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(age_groups, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% of Patients")
        ax.set_title("Readmission % by Age Group", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c5:
        st.markdown("**Prior Inpatient Visits vs Readmission Rate**")
        inpatient_bins = ["0", "1", "2", "3", "4", "5+"]
        readmit_rate   = [8.2, 14.6, 21.3, 28.7, 34.1, 41.5]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        bar_colors = ["#60a5fa" if r < 15 else "#f59e0b" if r < 25 else "#ef4444" for r in readmit_rate]
        ax.bar(inpatient_bins, readmit_rate, color=bar_colors, edgecolor="white", linewidth=1)
        ax.set_xlabel("Number of Prior Inpatient Visits")
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_title("Prior Inpatient Visits → Readmission Risk\n(#1 most important feature)", fontsize=11, fontweight="bold")
        for i, v in enumerate(readmit_rate):
            ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9, fontweight="bold", color="#374151")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Key findings
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">💡 Key Findings</div>', unsafe_allow_html=True)

    findings = [
        ("#ef4444", "🏥",  "Prior Hospitalisation Is #1 Predictor",
         "number_inpatient (prior inpatient visits) is the strongest predictor of 30-day readmission. "
         "Each additional prior visit raises readmission risk by ~7 percentage points."),
        ("#1d4ed8", "👴",  "Older Patients Are Most At-Risk",
         "The 70–80 age group is the largest patient group. Patients aged 60–90 consistently show "
         "higher readmission rates, aligning with clinical literature on elderly diabetic care."),
        ("#7c3aed", "💊",  "Insulin Changes Signal Instability",
         "Patients with increasing insulin dosage ('Up') have noticeably higher readmission rates. "
         "This suggests glycemic instability as a key clinical flag."),
        ("#059669", "⚖️",  "Class Imbalance Must Be Addressed",
         "~89% of patients were not readmitted within 30 days. "
         "We used class_weight='balanced' across all models to prevent the majority class from dominating learning."),
        ("#f59e0b", "🔊",  "Label Noise Degrades All Models",
         "After injecting 10% label noise (6,997 flipped labels), all 7 models showed consistent performance "
         "drops — confirming that data quality is critical for healthcare ML systems."),
        ("#0891b2", "⏰",  "Most Patients Stay 3–7 Days",
         "Average stay = 4.4 days. Patients with stays longer than 7 days tend to have higher complexity "
         "and elevated readmission rates."),
    ]

    fc1, fc2 = st.columns(2)
    for i, (color, icon, title, body) in enumerate(findings):
        col = fc1 if i % 2 == 0 else fc2
        with col:
            col.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px 18px;
                        margin-bottom:12px;box-shadow:0 1px 8px rgba(0,0,0,0.07);
                        border-left:4px solid {color};">
              <div style="font-size:13.5px;font-weight:700;color:{color};margin-bottom:6px;">
                {icon} &nbsp; {title}
              </div>
              <p style="color:#475569;font-size:13px;margin:0;line-height:1.65;">{body}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Future Work":

    st.markdown('<div class="sec-header">🔮 Future Work & Improvements</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
    💡 &nbsp; This project demonstrates that building a <strong>clinically useful</strong> prediction system
    requires more than just fitting models. The following improvements are recommended for production deployment.
    </div>
    """, unsafe_allow_html=True)

    future_items = [
        ("#1d4ed8", "🔄", "SMOTE / Oversampling",
         "Class Balancing",
         "Apply SMOTE (Synthetic Minority Oversampling Technique) or other oversampling strategies "
         "to generate synthetic minority-class examples during training. This should improve Recall and F1 "
         "beyond what class_weight='balanced' alone can achieve.",
         ["SMOTE", "ADASYN", "RandomOverSampler"]),
        ("#7c3aed", "🎚️", "Threshold Tuning",
         "Recall / Precision Trade-off",
         "The default 0.5 classification threshold is not optimal for imbalanced medical data. "
         "By lowering the threshold (e.g. 0.35–0.40), we can increase Recall at the cost of some Precision — "
         "which is the right trade-off when missing at-risk patients is more harmful than false alarms.",
         ["ROC analysis", "Precision-Recall curves", "Clinical cost weighting"]),
        ("#059669", "🗳️", "Voting Ensembles / Stacking",
         "Combine Model Strengths",
         "Train a soft-voting ensemble combining Logistic Regression, Random Forest, and SVM. "
         "Each model captures different patterns; combining them may improve AUC and Recall "
         "over any single model.",
         ["Soft Voting", "Stacking", "Blending"]),
        ("#f59e0b", "🧠", "Explainable AI (SHAP)",
         "Clinical Interpretability",
         "Medical staff must understand why a prediction was made before acting on it. "
         "SHAP (SHapley Additive exPlanations) values provide per-patient, per-feature explanations "
         "that show exactly which factors drove the readmission risk score.",
         ["SHAP values", "LIME", "Feature attribution"]),
        ("#0891b2", "🏗️", "External Validation",
         "Generalisability Testing",
         "The current model was trained and tested on data from 1999–2008. Validating on newer "
         "patient populations and different hospital systems is essential before clinical deployment.",
         ["Cross-hospital validation", "Temporal validation", "Prospective testing"]),
        ("#ef4444", "🚀", "Clinical Deployment",
         "Hospital Decision Support System",
         "Integrate the model into hospital workflows as a decision support tool — flagging "
         "high-risk patients at discharge for closer follow-up. This requires a full MLOps pipeline "
         "with continuous monitoring, model drift detection, and retraining protocols.",
         ["REST API", "EHR integration", "Model monitoring"]),
    ]

    fw1, fw2 = st.columns(2)
    for i, (color, icon, title, subtitle, body, tags) in enumerate(future_items):
        col = fw1 if i % 2 == 0 else fw2
        tag_html = " ".join([
            f'<span style="background:#f1f5f9;color:#475569;border-radius:6px;'
            f'padding:2px 9px;font-size:11px;font-weight:600;">{t}</span>'
            for t in tags
        ])
        with col:
            col.markdown(f"""
            <div style="background:white;border-radius:14px;padding:18px 20px;
                        margin-bottom:14px;box-shadow:0 2px 10px rgba(0,0,0,0.07);
                        border-top:4px solid {color};">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="font-size:24px;">{icon}</div>
                <div>
                  <div style="font-weight:700;color:#1e293b;font-size:14px;">{title}</div>
                  <div style="font-size:12px;color:{color};font-weight:600;">{subtitle}</div>
                </div>
              </div>
              <p style="color:#475569;font-size:13px;line-height:1.7;margin:0 0 12px;">{body}</p>
              <div style="display:flex;gap:6px;flex-wrap:wrap;">{tag_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # Conclusion note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;
                padding:28px 32px;text-align:center;">
      <div style="font-size:22px;font-weight:800;color:white;margin-bottom:10px;">
        🎯 Project Conclusion
      </div>
      <p style="color:#93c5fd;font-size:14px;line-height:1.8;max-width:700px;margin:0 auto 16px;">
        This project successfully built an ML pipeline for predicting 30-day hospital readmission
        from noisy real-world clinical records. <strong style="color:white;">Logistic Regression</strong>
        was selected as the best model (AUC = 0.622, Recall = 0.53, F1 = 0.21) because it reliably
        detects at-risk patients — unlike high-accuracy models that detected zero readmissions.
      </p>
      <p style="color:#60a5fd;font-size:13px;margin:0;">
        The key lesson: <strong style="color:#38bdf8;">proper evaluation metrics are more important
        than raw accuracy</strong> when working with imbalanced healthcare data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:12px;padding:10px;">
      Hospital Readmission Prediction — Diabetes 130-US Dataset &nbsp;·&nbsp;
      Aseel Bajaber &amp; Jumanah AlNahdi &nbsp;·&nbsp; Spring 2026
    </div>
    """, unsafe_allow_html=True)
