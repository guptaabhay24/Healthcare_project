# ============================================================
# app_diabetes.py — Streamlit Deployment for Diabetes Detection
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load model and preprocessor
# -------------------------------
MODEL_PATH = r"D:\Desktop\Python\healthcare_ml_project\projectenv\Diabetes_Project\models\final_LightGBM_tuned.pkl"
PREPROCESSOR_PATH = r"D:\Desktop\Python\healthcare_ml_project\projectenv\Diabetes_Project\models\preprocessor.pkl"

# Load pipeline (contains preprocessing + model)
pipe = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")

st.markdown("""
This app predicts the **probability of diabetes** based on key health metrics  
using your trained **LightGBM model** .
""")

# -------------------------------
# 2. User Input Section
# -------------------------------
st.header("Enter Patient Details:")

Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
Glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
SkinThickness = st.number_input("Skin Thickness (mm)", 0, 100, 25)
Insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
BMI = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 120, 35)

# Create dataframe for prediction
input_df = pd.DataFrame({
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
})

# -------------------------------
# 3. Make prediction
# -------------------------------
if st.button("🔍 Predict"):
    pred_prob = pipe.predict_proba(input_df)[0][1]
    pred_class = int(pred_prob >= 0.5)

    st.subheader("Prediction Result:")
    if pred_class == 1:
        st.error(f"⚠️ High Risk of Diabetes (Probability = {pred_prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability = {pred_prob:.2f})")

  

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with using Streamlit, LightGBM | Abhay’s Healthcare ML System")
