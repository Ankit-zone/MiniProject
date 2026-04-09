import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Health Risk Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ------------------ FEATURES ------------------
FEATURES = [
    "Age", "BMI", "Blood_Pressure", "Cholesterol",
    "Glucose_Level", "Heart_Rate", "Sleep_Hours",
    "Water_Intake", "Stress_Level"
]

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    import os
    path = os.path.join(os.path.dirname(__file__), "novagen_dataset.csv")
    df = pd.read_csv(path)
    return df[FEATURES]

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Dashboard Menu")
page = st.sidebar.radio("", ["🏠 Home", "📊 Analysis", "🤖 Prediction"])

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Ankit, Anurag, Aman 🚀")

# ------------------ HOME ------------------
if page == "🏠 Home":
    st.title("📊 Health Risk Prediction Dashboard")

    st.markdown("### 📌 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ------------------ ANALYSIS ------------------
elif page == "📊 Analysis":
    st.title("📊 Data Analysis & Visualization")

    st.markdown("### 🔍 Feature Relationships")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Select X-axis", FEATURES)

    with col2:
        y_axis = st.selectbox("Select Y-axis", FEATURES)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### 📈 Distribution")

    column = st.selectbox("Select Feature", FEATURES)

    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### 🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------ PREDICTION ------------------
elif page == "🤖 Prediction":
    st.title("🤖 Health Risk Prediction")

    model = pickle.load(open("model.pkl", "rb"))

    st.markdown("### 📝 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100, 25)
        bmi = st.slider("BMI", 10.0, 40.0, 22.0)
        bp = st.slider("Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 300, 180)
        glucose = st.slider("Glucose Level", 70, 200, 100)

    with col2:
        hr = st.slider("Heart Rate", 50, 120, 72)
        sleep = st.slider("Sleep Hours", 0, 12, 7)
        water = st.slider("Water Intake (glasses)", 0, 10, 5)
        stress = st.slider("Stress Level", 0, 10, 5)

    input_data = [[
        age, bmi, bp, chol, glucose,
        hr, sleep, water, stress
    ]]

    st.markdown("---")

    if st.button("🚀 Predict", use_container_width=True):
        with st.spinner("Analyzing..."):
            prediction = model.predict(input_data)

        st.success(f"✅ Prediction Result: {prediction[0]}")
