<div align="center">

# 🏥 Health Risk Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

### 🚀 Predict patient health risk in real time with an interactive ML-powered dashboard

[![Live App](https://img.shields.io/badge/🌐_Live_App-Click_Here-00e5ff?style=for-the-badge)](https://your-app-link.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/📁_GitHub-Source_Code-181717?style=for-the-badge)](https://github.com/YOUR_USERNAME/HealthRiskPrediction)

</div>

---

## 📌 Overview

**Health Risk Prediction Dashboard** is an end-to-end Machine Learning project that predicts whether a patient is at **High Risk** or **Low Risk** of a health condition — based on 9 key medical and lifestyle attributes.

Early identification of at-risk patients can help healthcare providers **intervene sooner**, reduce hospitalizations, and **save lives**. This dashboard makes that prediction instant, visual, and accessible to anyone.

> Built entirely from scratch — raw medical dataset → data preprocessing → model training → interactive 3-page web dashboard → public deployment.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🏠 **Home Page** | Dataset overview with total records, features, and missing values |
| 📋 **Dataset Preview** | Live scrollable table showing first rows of the dataset |
| 📊 **Scatter Plot** | Interactive X vs Y feature relationship explorer |
| 📈 **Distribution Plot** | Histogram with KDE curve for any selected feature |
| 🔥 **Correlation Heatmap** | Full feature correlation matrix with color coding |
| 🤖 **Real-time Prediction** | Predicts High/Low health risk from 9 patient inputs |
| 🎛️ **Interactive Sliders** | Easy input for all 9 medical parameters |
| ✅ **Clear Results** | Color-coded result — green for Low Risk, red for High Risk |

---

## 🖥️ Dashboard Pages

```
┌─────────────────────────────────────────────────────────┐
│  📌 Dashboard Menu                                       │
│  ○ 🏠 Home                                              │
│  ○ 📊 Analysis                                          │
│  ○ 🤖 Prediction                                        │
└─────────────────────────────────────────────────────────┘

🏠 HOME PAGE
┌──────────────┬──────────────┬──────────────┐
│ Total Records│Total Features│Missing Values│
│    9,549     │      9       │      0       │
└──────────────┴──────────────┴──────────────┘
│  Dataset Preview Table                      │

📊 ANALYSIS PAGE
│  Feature Scatter Plot  (X vs Y selector)    │
│  Distribution Histogram (KDE curve)         │
│  Correlation Heatmap   (10x5 coolwarm)      │

🤖 PREDICTION PAGE
┌─────────────────┬──────────────────────────┐
│ Age    [slider] │ Heart Rate      [slider] │
│ BMI    [slider] │ Sleep Hours     [slider] │
│ BP     [slider] │ Water Intake    [slider] │
│ Chol   [slider] │ Stress Level    [slider] │
│ Glucose[slider] │                          │
└─────────────────┴──────────────────────────┘
│  🚀 Predict Button                          │
│  ✅ Low Health Risk / ⚠️ High Health Risk   │
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Model** | Scikit-learn — Logistic Regression (L2, lbfgs solver) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard UI** | Streamlit |
| **Model Serialization** | Pickle |
| **Version Control** | Git & GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📊 Dataset — Novagen Health Dataset

| Property | Value |
|---|---|
| **File** | `novagen_dataset.csv` |
| **Total Rows** | 9,549 patients |
| **Total Columns** | 23 (9 used for prediction) |
| **Missing Values** | ✅ None |
| **Target Classes** | 0 = Low Risk · 1 = High Risk |
| **Class Distribution** | Low Risk: 4,570 · High Risk: 4,979 |

### Features Used for Prediction

| Feature | Type | Range | Description |
|---|---|---|---|
| **Age** | Numeric | 1 – 100 | Patient age in years |
| **BMI** | Numeric | 10 – 40 | Body Mass Index |
| **Blood_Pressure** | Numeric | 80 – 200 | Systolic blood pressure (mmHg) |
| **Cholesterol** | Numeric | 100 – 300 | Cholesterol level (mg/dL) |
| **Glucose_Level** | Numeric | 70 – 200 | Blood glucose level (mg/dL) |
| **Heart_Rate** | Numeric | 50 – 120 | Resting heart rate (bpm) |
| **Sleep_Hours** | Numeric | 0 – 12 | Average daily sleep hours |
| **Water_Intake** | Numeric | 0 – 10 | Daily water intake (glasses) |
| **Stress_Level** | Numeric | 0 – 10 | Self-reported stress level |

### Additional Dataset Columns (not used in prediction)

`Smoking`, `Alcohol`, `Diet`, `MentalHealth`, `PhysicalActivity`, `MedicalHistory`, `Allergies`, `Diet_Type`, `Blood_Group`

---

## 🔑 Key Insights from the Dataset

- 📌 Dataset is **nearly balanced** — 47.9% Low Risk vs 52.1% High Risk
- 📌 **Zero missing values** — no imputation required
- 📌 **Stress Level + Blood Pressure** show strong correlation with health risk
- 📌 **Low sleep hours** combined with **high stress** is a significant risk signal
- 📌 **High glucose + high cholesterol** together strongly indicate High Risk

---

## 🤖 Model Details

| Property | Value |
|---|---|
| **Algorithm** | Logistic Regression |
| **Penalty** | L2 (Ridge) |
| **Solver** | lbfgs |
| **Max Iterations** | 100 |
| **C (Regularization)** | 1.0 |
| **Classes** | 0 = Low Risk, 1 = High Risk |
| **Input Features** | 9 |
| **Saved As** | `model.pkl` |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/HealthRiskPrediction.git
cd HealthRiskPrediction
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the dashboard
```bash
streamlit run app.py
```

### Step 5 — Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
HealthRiskPrediction/
│
├── app.py                   # Streamlit dashboard (3-page app)
├── model.pkl                # Trained Logistic Regression model
├── novagen_dataset.csv      # Novagen health dataset (9,549 rows)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🌐 Live Deployment

The app is deployed on **Streamlit Community Cloud** and accessible publicly:

🔗 **[Click here to open the live app](https://your-app-link.streamlit.app)**

No installation needed — just open the link and start exploring!

---

## 👥 Team

This project was built by a team of 3:

| Name | Role |
|---|---|
| **Ankit** | ML Model + Prediction Page + Deployment |
| **Anurag** | Data Analysis + Visualization Page |
| **Aman** | Data Preprocessing + Home Page + UI |

---

## 🙋 Connect With Us

**Ankit**
- 🌐 LinkedIn: [www.linkedin.com/in/iankityadav03]
- 💻 GitHub: [https://github.com/Ankit-zone]

---

## ⭐ Support

If you found this project useful, please consider giving it a **star ⭐** on GitHub — it helps others find it too!

---

<div align="center">

Made with ❤️ by Ankit, Anurag & Aman &nbsp;|&nbsp; Built with Python & Streamlit

</div>
