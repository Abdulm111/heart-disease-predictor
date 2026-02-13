import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models & Scaler
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "KNN": "models/knn_model.pkl",
        "Naive Bayes": "models/naive_bayes_model.pkl",
        "Logistic Regression": "models/logistic_regression_model.pkl",
        "SVM": "models/svm_model.pkl",
        "Random Forest": "models/random_forest_model.pkl",
        "XGBoost": "models/xgboost_model.pkl"
    }
    for name, path in model_files.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

@st.cache_resource
def load_scaler():
    with open("data/preprocessed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data["scaler"]

models = load_models()
scaler = load_scaler()

# ---------- Header ----------
st.markdown("""
    <div style='text-align:center; padding: 20px;'>
        <h1>🏥 Heart Disease Predictor</h1>
        <p style='font-size:18px; color: gray;'>
            Enter patient details and let AI predict the risk of heart disease
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- Sidebar: Patient Input ----------
st.sidebar.header("🩺 Patient Information")
st.sidebar.markdown("---")

age = st.sidebar.slider("📅 Age (years)", min_value=29, max_value=100, value=50, step=1)

gender = st.sidebar.selectbox("🙎‍♂️ Gender", ["Male", "Female"])
gender_val = 2 if gender == "Male" else 1

height = st.sidebar.slider("📏 Height (cm)", min_value=120, max_value=210, value=165, step=1)
weight = st.sidebar.slider("⚖️ Weight (kg)", min_value=30, max_value=200, value=70, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🩸 Blood Pressure")
ap_hi = st.sidebar.slider("Systolic (ap_hi)", min_value=80, max_value=200, value=120, step=1)
ap_lo = st.sidebar.slider("Diastolic (ap_lo)", min_value=50, max_value=140, value=80, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Medical Tests")
cholesterol = st.sidebar.selectbox("🧪 Cholesterol Level", [1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
gluc = st.sidebar.selectbox("🍬 Glucose Level", [1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
st.sidebar.caption("ℹ️ Cholesterol & Glucose levels are categorized as per the dataset. Actual lab values are not supported.")

st.sidebar.markdown("---")
st.sidebar.subheader("🏃 Lifestyle")
smoke = st.sidebar.selectbox("🚬 Smoking", ["No", "Yes"])
smoke_val = 1 if smoke == "Yes" else 0

alco = st.sidebar.selectbox("🍺 Alcohol Intake", ["No", "Yes"])
alco_val = 1 if alco == "Yes" else 0

active = st.sidebar.selectbox("🏃 Physical Activity", ["No", "Yes"])
active_val = 1 if active == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
selected_model = st.sidebar.selectbox("Choose ML Model", list(models.keys()))

model_accuracies = {
    "KNN": 69.32,
    "Naive Bayes": 70.78,
    "Logistic Regression": 72.74,
    "SVM": 73.26,
    "Random Forest": 70.78,
    "XGBoost": 73.01
}
st.sidebar.info(f"📊 {selected_model} Accuracy: {model_accuracies[selected_model]}%")

# ---------- Calculate BMI ----------
bmi = weight / ((height / 100) ** 2)

# ---------- BMI Category ----------
def get_bmi_category(bmi_val):
    if bmi_val < 18.5:
        return "🔵 Underweight", "info"
    elif bmi_val < 25:
        return "🟢 Normal", "success"
    elif bmi_val < 30:
        return "🟡 Overweight", "warning"
    else:
        return "🔴 Obese", "error"

bmi_category, bmi_status = get_bmi_category(bmi)

# ---------- Pre-Prediction: Show Current Patient Stats ----------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Age", f"{age} years")
with col2:
    st.metric("⚖️ BMI", f"{round(bmi, 1)}", delta=bmi_category)
with col3:
    st.metric("🩸 Blood Pressure", f"{ap_hi}/{ap_lo} mmHg")
with col4:
    st.metric("🤖 Model", selected_model)
if age < 29 or age > 64:
    st.warning(f"⚠️ The model was trained on ages 29-64. Age {age} is outside this range — prediction may be less accurate.")

# ---------- Build Feature Array ----------
features = np.array([[
    age, gender_val, height, weight, ap_hi, ap_lo,
    cholesterol, gluc, smoke_val, alco_val, active_val, bmi
]])
features_scaled = scaler.transform(features)

# ---------- Predict Button ----------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Heart Disease", use_container_width=True, type="primary")

# ---------- Prediction Result ----------
if predict_btn:
    model = models[selected_model]
    prediction = model.predict(features_scaled)
    st.markdown("---")

    if prediction[0] == 1:
        st.markdown("""
            <div style='text-align:center; padding:30px; background-color:#ff4b4b22; border-radius:15px; border: 2px solid #ff4b4b;'>
                <h1>⚠️ Heart Disease Detected!</h1>
                <p style='font-size:20px;'>The model predicts this patient is at <b>RISK</b> of heart disease.</p>
                <p style='font-size:16px; color:gray;'>Please consult a cardiologist for further evaluation.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='text-align:center; padding:30px; background-color:#00cc4422; border-radius:15px; border: 2px solid #00cc44;'>
                <h1>✅ No Heart Disease Detected!</h1>
                <p style='font-size:20px;'>The model predicts this patient is <b>HEALTHY</b>.</p>
                <p style='font-size:16px; color:gray;'>Maintain a healthy lifestyle for continued well-being!</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BMI Status ----------
    st.subheader("🏋️ BMI Analysis")
    if bmi_status == "info":
        st.info(f"BMI: {round(bmi, 2)} — {bmi_category} | Consider a balanced diet to gain healthy weight.")
    elif bmi_status == "success":
        st.success(f"BMI: {round(bmi, 2)} — {bmi_category} | Great! Your BMI is in the healthy range!")
    elif bmi_status == "warning":
        st.warning(f"BMI: {round(bmi, 2)} — {bmi_category} | Consider regular exercise and a balanced diet.")
    else:
        st.error(f"BMI: {round(bmi, 2)} — {bmi_category} | High BMI increases heart disease risk. Please consult a doctor.")

    # ---------- Patient Summary ----------
    st.subheader("📋 Patient Summary")
    chol_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    gluc_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}

    patient_data = {
        "Feature": [
            "Age", "Gender", "Height", "Weight", "BMI",
            "Systolic BP", "Diastolic BP", "Cholesterol",
            "Glucose", "Smoking", "Alcohol Intake", "Physical Activity"
        ],
        "Value": [
            f"{age} years", gender, f"{height} cm", f"{weight} kg", round(bmi, 2),
            f"{ap_hi} mmHg", f"{ap_lo} mmHg", chol_map[cholesterol],
            gluc_map[gluc], smoke, alco, active
        ]
    }
    df = pd.DataFrame(patient_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # ---------- Model Comparison ----------
    st.subheader("📊 Model Comparison")
    models_list = list(model_accuracies.items())
    best_model = max(model_accuracies, key=model_accuracies.get)

    for i in range(0, len(models_list), 3):
        cols = st.columns(3)
        for col, (model_name, acc) in zip(cols, models_list[i:i+3]):
            with col:
                if model_name == selected_model:
                    st.metric(label=f"✅ {model_name} (Selected)", value=f"{acc}%")
                elif model_name == best_model:
                    st.metric(label=f"👑 {model_name}", value=f"{acc}%", delta="Best!")
                else:
                    st.metric(label=model_name, value=f"{acc}%")

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:gray; font-size:14px;'>
        Made with ❤️ | Heart Disease Predictor | Machine Learning Project | 2025
    </p>
""", unsafe_allow_html=True)