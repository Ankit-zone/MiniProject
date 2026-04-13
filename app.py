import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import plotly.express as px

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
    path = os.path.join(os.path.dirname(__file__), "novagen_dataset.csv")
    df = pd.read_csv(path)
    return df[FEATURES]

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Dashboard Menu")
page = st.sidebar.radio("", ["🏠 Home", "📊 Analysis", "🤖 Prediction"])

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Ankit 🚀")

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
    st.title("📊 Interactive Data Dashboard")

    tab1, tab2 = st.tabs(["📊 Overview Dashboard", "📈 Advanced Dashboard"])

    # ---------------- DASHBOARD 1 ----------------
    with tab1:
        st.subheader("📊 Overview Dashboard")

        col1, col2 = st.columns(2)

        # Histogram
        with col1:
            feature = st.selectbox("Select Feature (Histogram)", FEATURES)
            fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}")
            st.plotly_chart(fig, use_container_width=True)

        # Box Plot
        with col2:
            fig = px.box(df, y=feature, title=f"Box Plot of {feature}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Scatter Plot
        col3, col4 = st.columns(2)

        with col3:
            x_axis = st.selectbox("X-axis", FEATURES, key="x1")

        with col4:
            y_axis = st.selectbox("Y-axis", FEATURES, key="y1")

        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            title=f"{x_axis} vs {y_axis}",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.subheader("🔥 Correlation Heatmap")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- DASHBOARD 2 ----------------
    with tab2:
    st.subheader("📈 Simple Insights Dashboard")

    # ---------------- TOP SECTION ----------------
    st.markdown("### 📊 Key Feature Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="BMI", title="BMI Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Blood_Pressure", title="Blood Pressure Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------- MIDDLE SECTION ----------------
    st.markdown("### 🔍 Relationship Between Important Features")

    fig = px.scatter(
        df,
        x="BMI",
        y="Blood_Pressure",
        title="BMI vs Blood Pressure",
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------- BOTTOM SECTION ----------------
    st.markdown("### 📌 Select Any Feature to Explore")

    feature = st.selectbox("Choose Feature", FEATURES)

    fig = px.box(df, y=feature, title=f"{feature} Overview")
    st.plotly_chart(fig, use_container_width=True)
# ------------------ PREDICTION ------------------
elif page == "🤖 Prediction":
    st.title("🤖 Health Risk Prediction")

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = pickle.load(open(model_path, "rb"))

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

        # ---------------- ML Prediction ----------------
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]

        # ---------------- RULE-BASED LOGIC ----------------
        if bmi >= 30 or bp >= 140:
            rule_risk = "High Risk"
            st.error("🔴 Rule-Based: High Health Risk")
        elif (25 <= bmi < 30) or (120 <= bp < 140):
            rule_risk = "Moderate Risk"
            st.warning("🟡 Rule-Based: Moderate Risk")
        else:
            rule_risk = "Low Risk"
            st.success("🟢 Rule-Based: Low Risk")

        st.markdown("---")

        # ---------------- ML RESULT ----------------
        if prediction[0] == 1:
            st.error(f"🤖 ML Prediction: High Risk ({proba*100:.2f}%)")
        else:
            st.success(f"🤖 ML Prediction: Low Risk ({(1-proba)*100:.2f}%)")

        # ---------------- FINAL METRIC ----------------
        st.metric("Final Risk Level", rule_risk)
