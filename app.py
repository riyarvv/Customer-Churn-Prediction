import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")

# Sidebar inputs
st.sidebar.header("User Input Features")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0)

# Predict button
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([[gender, tenure, monthly_charges]],
                            columns=["gender", "tenure", "MonthlyCharges"])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: **{'Churn' if prediction == 1 else 'No Churn'}**")
