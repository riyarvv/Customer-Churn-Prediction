import streamlit as st
import pandas as pd
import pickle

with open("best_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.sidebar.header("User Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0)

def preprocess_input(gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges):
    # Convert categorical to numeric exactly like training
    gender = 1 if gender == "Male" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    
    # Create DataFrame in the exact column order the model was trained on
    input_data = pd.DataFrame([[
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure,
        monthly_charges,
        total_charges
    ]], columns=[
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "MonthlyCharges", "TotalCharges"
    ])
    return input_data

if st.sidebar.button("Predict Churn"):
    input_df = preprocess_input(gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges)
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)


