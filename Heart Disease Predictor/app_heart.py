import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load model and scaler
model = joblib.load(r"D:\Desktop\Python\healthcare_ml_project\projectenv\etc\jupyter\heart_model.pkl")
scaler = joblib.load(r"D:\Desktop\Python\healthcare_ml_project\projectenv\etc\jupyter\heart_scaler.pkl")

# Load original dataset to get realistic ranges
df = pd.read_csv(r"D:\Desktop\Python\healthcare_ml_project\heart.csv")

st.title("❤️ Heart Disease Risk Predictor with Probabilistic Score & Explainability")

# --- Feature columns ---
columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal']

# --- Dynamic sliders based on dataset min/max ---
age = st.slider("Age", int(df.age.min()), int(df.age.max()), int(df.age.median()))
sex_display = st.selectbox("Sex", ["F", "M"])
sex = 0 if sex_display == "F" else 1

cp_display = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"]
)
cp = int(cp_display.split("(")[1][0])

trestbps = st.slider("Resting BP (mm Hg)", int(df.trestbps.min()), int(df.trestbps.max()), int(df.trestbps.median()))
chol = st.slider("Cholesterol (mg/dl)", int(df.chol.min()), int(df.chol.max()), int(df.chol.median()))

fbs_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False (0)", "True (1)"])
fbs = int(fbs_display.split("(")[1][0])

restecg_display = st.selectbox(
    "Resting ECG",
    ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"]
)
restecg = int(restecg_display.split("(")[1][0])

thalach = st.slider("Max Heart Rate Achieved", int(df.thalach.min()), int(df.thalach.max()), int(df.thalach.median()))

exang_display = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
exang = int(exang_display.split("(")[1][0])

oldpeak = st.slider("ST Depression (oldpeak)", float(df.oldpeak.min()), float(df.oldpeak.max()), float(df.oldpeak.median()), step=0.1)

slope_display = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping (0)", "Flat (1)", "Downsloping (2)"]
)
slope = int(slope_display.split("(")[1][0])

ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0,1,2,3])

thal_display = st.selectbox(
    "Thalassemia",
    ["Fixed Defect (1)", "Normal (2)", "Reversible Defect (3)"]
)
thal = int(thal_display.split("(")[1][0])

# --- Prepare input DataFrame ---
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=columns)

# Scale
input_scaled = scaler.transform(input_data)

# --- Predict button ---
if st.button("Predict"):

    # Probabilistic prediction
    prob = model.predict_proba(input_scaled)[0][1] * 100  # probability of disease
    st.write(f"⚠️ Estimated Risk of Heart Disease: **{prob:.2f}%**")

    # Binary label based on 50% threshold
    pred = model.predict(input_scaled)[0]
    if pred == 1:
        st.error("💔 High Risk of Heart Disease")
    else:
        st.success("❤️ No Heart Disease Risk")

    # --- Explainability using SHAP ---
    explainer = shap.Explainer(model, scaler.transform(df[columns]))  # explain using scaled dataset
    shap_values = explainer(input_scaled)
    
   
