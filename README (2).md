# 🩺 MediPredict AI — Disease Risk Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered web application that predicts diabetes risk from patient health data using Gradient Boosting ML — achieving 89% accuracy and 95% AUC-ROC score.**

[🚀 Live Demo](#) &nbsp;·&nbsp; [📊 Model Details](#-model-details) &nbsp;·&nbsp; [🛠️ Installation](#️-installation) &nbsp;·&nbsp; [📁 Project Structure](#-project-structure)

</div>

---

## 📸 Screenshots

> *Interactive web app with real-time risk prediction, gauge chart, feature importance and personalised health recommendations*

---

## ✨ Key Features

- 🎯 **Real-time risk prediction** — instant diabetes risk score from 8 health parameters
- 📊 **Interactive gauge chart** — visual risk meter with colour-coded zones (low / moderate / high)
- 🔍 **Feature importance chart** — understand which health factors drive your risk the most
- 💡 **Personalised recommendations** — tailored health advice based on risk level
- 📋 **Patient summary table** — compare your values vs. normal clinical ranges
- ⚡ **Fast & lightweight** — loads in under 2 seconds, no login required

---

## 🧠 Model Details

| Metric | Score |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Accuracy | **89.25%** |
| AUC-ROC | **95.45%** |
| Cross-Val Score (5-fold) | **90.00%** |
| Training Samples | 2,000 |
| Features Used | 8 |

### 📥 Input Features

| Feature | Description | Normal Range |
|---|---|---|
| Age | Patient age in years | — |
| BMI | Body Mass Index | 18.5 – 24.9 |
| Fasting Glucose | Blood glucose (mg/dL) | < 100 |
| Blood Pressure | Systolic BP (mmHg) | < 120 |
| HbA1c | 3-month avg blood sugar (%) | < 5.7% |
| Family History | Diabetes in immediate family | — |
| Physical Activity | Weekly exercise level (0–3) | Moderate+ |
| Smoking | Current smoking status | Non-smoker |

### 🏆 Feature Importance (Top Predictors)

```
Glucose Level      ████████████████████  29.2%
Age                ██████████████        18.5%
HbA1c              ████████████          16.1%
BMI                ██████████            14.0%
Family History     ██████                 9.2%
Blood Pressure     █████                  7.6%
Physical Activity  ██                     3.3%
Smoking            █                      2.0%
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MohammadMaaz6229/disease-risk-predictor.git
cd disease-risk-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🚀

---

## 📁 Project Structure

```
disease-risk-predictor/
│
├── app.py                    # Streamlit web application (main)
├── train_model.py            # Model training & evaluation script
├── requirements.txt          # Python dependencies
├── diabetes_data.csv         # Synthetic training dataset (2,000 rows)
├── model.pkl                 # Trained Gradient Boosting model
├── feature_importance.json   # Feature importance scores
└── README.md                 # Project documentation
```

---

## 🔬 Technical Approach

```
Raw Health Data  ──►  Data Generation  ──►  Feature Engineering
                                                    │
                                                    ▼
                                         StandardScaler (normalise)
                                                    │
                                                    ▼
                                    GradientBoostingClassifier
                                    (150 estimators, depth=4)
                                                    │
                                                    ▼
                                         Risk Probability Score
                                                    │
                                                    ▼
                              Streamlit UI ──► Personalised Report
```

---

## 🚀 Future Improvements

- [ ] Add support for Heart Disease & Hypertension prediction
- [ ] Integrate real-world NHS / Kaggle PIMA dataset
- [ ] Add SHAP (SHapley Additive exPlanations) for model explainability
- [ ] Deploy on AWS EC2 / Streamlit Cloud with CI/CD pipeline
- [ ] Add PDF report download for patients
- [ ] REST API using FastAPI for integration with hospital systems

---

## ⚠️ Disclaimer

This application is built for **educational and portfolio purposes only**. It does **not** constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 👨‍💻 Author

**Mohammad Maaz**
- 🎓 MCA Final Year — IGNOU
- 💼 Aspiring AI/ML Engineer & Data Scientist
- 🌍 Open to opportunities in India 🇮🇳 · Canada 🇨🇦 · Germany 🇩🇪

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mohd-maaz-534012235)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/MohammadMaaz6229)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify and distribute.

---

<div align="center">
⭐ If this project helped you, please consider giving it a star!
</div>
