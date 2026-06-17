import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI — Disease Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model & feature importances ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_feature_importance():
    with open("feature_importance.json") as f:
        return json.load(f)

model = load_model()
feat_imp = load_feature_importance()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 50%, #118ab2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.5rem; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0; }

.metric-card {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .num  { font-size: 2rem; font-weight: 700; color: #1b6ca8; }
.metric-card .lbl  { font-size: 0.8rem; color: #6b7a8d; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

.risk-high   { background: linear-gradient(135deg,#ff6b6b,#ee5a24); color:white; border-radius:12px; padding:1.5rem; text-align:center; }
.risk-medium { background: linear-gradient(135deg,#ffd32a,#f79f1f); color:white; border-radius:12px; padding:1.5rem; text-align:center; }
.risk-low    { background: linear-gradient(135deg,#05c46b,#0be881); color:white; border-radius:12px; padding:1.5rem; text-align:center; }
.risk-title  { font-size:1.6rem; font-weight:700; margin-bottom:0.5rem; }
.risk-pct    { font-size:3rem; font-weight:800; }
.risk-sub    { font-size:0.9rem; opacity:0.9; margin-top:0.5rem; }

.tip-card {
    background: #f0f7ff;
    border-left: 4px solid #1b6ca8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #1a2d45;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a2d45;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e8ecf0;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🩺 MediPredict AI</h1>
  <p>AI-powered Diabetes Risk Predictor &nbsp;·&nbsp; Built with Gradient Boosting ML &nbsp;·&nbsp; 89% Accuracy · 95% AUC-ROC</p>
</div>
""", unsafe_allow_html=True)

# ── Model metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in zip(
    [c1,c2,c3,c4],
    ["89%","95%","2,000","8"],
    ["Model Accuracy","AUC-ROC Score","Training Samples","Health Features"]
):
    col.markdown(f'<div class="metric-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Health Data")
    st.markdown("Fill in the details below to get your risk assessment.")
    st.divider()

    st.markdown("**👤 Personal Info**")
    age = st.slider("Age (years)", 18, 90, 45)
    bmi = st.slider("BMI", 15.0, 55.0, 26.0, 0.1,
                    help="Body Mass Index = weight(kg) / height(m)²")

    st.divider()
    st.markdown("**🩸 Clinical Measurements**")
    glucose = st.slider("Fasting Glucose (mg/dL)", 70, 300, 100,
                        help="Normal: <100 · Pre-diabetic: 100–125 · Diabetic: >126")
    bp = st.slider("Blood Pressure (mmHg)", 60, 160, 80,
                   help="Systolic blood pressure")
    hba1c = st.slider("HbA1c (%)", 4.0, 12.0, 5.5, 0.1,
                      help="3-month avg blood sugar. Normal: <5.7% · Pre-diabetic: 5.7–6.4% · Diabetic: >6.5%")

    st.divider()
    st.markdown("**🏃 Lifestyle Factors**")
    family_history = st.selectbox("Family History of Diabetes", ["No","Yes"])
    activity = st.selectbox("Physical Activity Level",
                            ["None (Sedentary)","Low (1–2x/week)","Moderate (3–4x/week)","High (5+/week)"])
    smoking = st.selectbox("Smoking Status", ["Non-Smoker","Smoker"])

    st.divider()
    predict_btn = st.button("🔍 Predict My Risk", use_container_width=True, type="primary")

# ── Map inputs ────────────────────────────────────────────────────────────────
activity_map = {
    "None (Sedentary)": 0,
    "Low (1–2x/week)": 1,
    "Moderate (3–4x/week)": 2,
    "High (5+/week)": 3
}
features = np.array([[
    age, bmi, glucose, bp, hba1c,
    1 if family_history == "Yes" else 0,
    activity_map[activity],
    1 if smoking == "Smoker" else 0
]])
feature_names = ["age","bmi","glucose_level","blood_pressure","hba1c",
                 "family_history","physical_activity_level","smoking"]

# ── Main content ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">🎯 Risk Assessment</div>', unsafe_allow_html=True)

    if predict_btn:
        prob = model.predict_proba(features)[0][1]
        pct  = round(prob * 100, 1)

        if prob >= 0.65:
            risk_cls, risk_label, emoji = "risk-high", "HIGH RISK", "🔴"
            tips = [
                "🏥 Consult a doctor or endocrinologist immediately",
                "🥗 Switch to a low-glycaemic diet (reduce sugar & refined carbs)",
                "🏃 Begin at least 150 min/week of moderate exercise",
                "💊 Discuss medication options with your physician",
                "📅 Schedule HbA1c & fasting glucose tests every 3 months"
            ]
        elif prob >= 0.35:
            risk_cls, risk_label, emoji = "risk-medium", "MODERATE RISK", "🟡"
            tips = [
                "👨‍⚕️ Visit your doctor for a full diabetes screening",
                "🥦 Increase vegetables, fibre & whole grains in your diet",
                "🏃 Aim for 30 minutes of walking or exercise daily",
                "⚖️ Losing 5–7% body weight significantly reduces risk",
                "📅 Check blood sugar every 6 months"
            ]
        else:
            risk_cls, risk_label, emoji = "risk-low", "LOW RISK", "🟢"
            tips = [
                "✅ Great news — keep up your healthy habits!",
                "🥗 Maintain a balanced diet rich in whole foods",
                "🏃 Stay active — 150 min/week of exercise is ideal",
                "🚭 Avoid smoking and limit alcohol",
                "📅 Annual health checkups are still recommended"
            ]

        st.markdown(f"""
        <div class="{risk_cls}">
          <div class="risk-title">{emoji} {risk_label}</div>
          <div class="risk-pct">{pct}%</div>
          <div class="risk-sub">Predicted probability of diabetes risk</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 32, "color": "#1a2d45"}},
            gauge={
                "axis": {"range": [0,100], "tickwidth": 1, "tickcolor": "#aaa"},
                "bar": {"color": "#1b6ca8", "thickness": 0.25},
                "steps": [
                    {"range": [0,35],   "color": "#d4edda"},
                    {"range": [35,65],  "color": "#fff3cd"},
                    {"range": [65,100], "color": "#f8d7da"}
                ],
                "threshold": {"line": {"color": "#1b6ca8","width": 4}, "value": pct}
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=20,b=0,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)", font_color="#1a2d45")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<div class="section-title">💡 Personalised Recommendations</div>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Fill in your health data on the left and click **Predict My Risk** to get your personalised assessment.")
        st.markdown("""
        **What this tool predicts:**
        - Likelihood of diabetes risk based on your health profile
        - Personalised recommendations based on your risk level
        - Key factors driving your risk score

        **⚠️ Disclaimer:** This tool is for educational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider.
        """)

with right:
    st.markdown('<div class="section-title">📊 Feature Importance</div>', unsafe_allow_html=True)

    labels = {
        "glucose_level": "Glucose Level",
        "age": "Age",
        "hba1c": "HbA1c",
        "bmi": "BMI",
        "family_history": "Family History",
        "blood_pressure": "Blood Pressure",
        "physical_activity_level": "Physical Activity",
        "smoking": "Smoking"
    }
    sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1])
    fig_fi = go.Figure(go.Bar(
        x=[v for _,v in sorted_fi],
        y=[labels[k] for k,_ in sorted_fi],
        orientation="h",
        marker=dict(
            color=[v for _,v in sorted_fi],
            colorscale=[[0,"#cce5ff"],[1,"#0f4c75"]],
            showscale=False
        ),
        text=[f"{v*100:.1f}%" for _,v in sorted_fi],
        textposition="outside"
    ))
    fig_fi.update_layout(
        height=300, margin=dict(t=10,b=10,l=10,r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    if predict_btn:
        st.markdown('<div class="section-title">📋 Your Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Health Parameter": [
                "Age", "BMI", "Fasting Glucose", "Blood Pressure",
                "HbA1c", "Family History", "Physical Activity", "Smoking"
            ],
            "Your Value": [
                f"{age} yrs", f"{bmi}", f"{glucose} mg/dL", f"{bp} mmHg",
                f"{hba1c}%", family_history, activity, smoking
            ],
            "Normal Range": [
                "—", "18.5–24.9", "<100 mg/dL", "<120 mmHg",
                "<5.7%", "—", "Moderate+", "Non-Smoker"
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">📈 Risk Scale Reference</div>', unsafe_allow_html=True)
    risk_ref = pd.DataFrame({
        "Risk Level": ["🟢 Low", "🟡 Moderate", "🔴 High"],
        "Probability": ["0 – 35%", "35 – 65%", "65 – 100%"],
        "Recommended Action": [
            "Annual checkup, maintain lifestyle",
            "Doctor visit, lifestyle changes",
            "Immediate medical consultation"
        ]
    })
    st.dataframe(risk_ref, hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#6b7a8d; font-size:0.85rem; padding: 1rem 0;">
  Built by <strong>Mohammad Maaz</strong> &nbsp;·&nbsp;
  MCA Final Year @ IGNOU &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/mohd-maaz-534012235" target="_blank">LinkedIn</a> &nbsp;·&nbsp;
  <a href="https://github.com/MohammadMaaz6229" target="_blank">GitHub</a><br><br>
  ⚠️ <em>This tool is for educational purposes only. Not a substitute for professional medical advice.</em>
</div>
""", unsafe_allow_html=True)
