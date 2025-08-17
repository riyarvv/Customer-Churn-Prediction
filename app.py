import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("best_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction & Insights")

# Sidebar inputs
st.sidebar.header("User Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=0.1)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=10.0)

# Data preprocessing for model
input_data = {
    'gender': [gender],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless],
    'PaymentMethod': [payment],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
}

input_df = pd.DataFrame(input_data)

# Encode categoricals the same way as training
for col in input_df.select_dtypes(include=['object']).columns:
    input_df[col] = pd.Categorical(input_df[col]).codes

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

# EDA / Visualizations
st.markdown("---")
st.header("📈 Customer Churn Insights")

# Load your dataset (same one used for training, keep in repo as customer_churn.csv)
@st.cache_data
def load_data():
    return pd.read_csv("customer_churn.csv")

df = load_data()

# Pie chart: Churn distribution
fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], labels=["No Churn", "Churn"], ax=ax1)
ax1.set_ylabel("")
st.subheader("Churn Distribution")
st.pyplot(fig1)

# Bar chart: Churn by Contract
fig2, ax2 = plt.subplots()
sns.barplot(x="Contract", y="Churn", data=df, ax=ax2, ci=None, palette="Set2")
st.subheader("Churn Rate by Contract Type")
st.pyplot(fig2)

# Boxplot: Monthly charges vs churn
fig3, ax3 = plt.subplots()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3", ax=ax3)
st.subheader("Monthly Charges by Churn Status")
st.pyplot(fig3)



