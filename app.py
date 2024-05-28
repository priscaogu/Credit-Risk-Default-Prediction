import pickle
import pandas as pd
import streamlit as st
import sklearn
from sklearn.ensemble import RandomForestClassifier

# Function to load the model
def load_model():
    model_path = "cr_model.pkt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None

    try:
        with open(model_path, "rb") as file:
            rf = pickle.load(file)
        return rf
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to convert home ownership to numerical value
def home_ownership(x):
    if x == 'OWN':
        return 0
    elif x == 'MORTGAGE':
        return 1
    elif x == 'RENT':
        return 2
    else:
        return 3

# Function to convert person default to numerical value
def person_default(x):
    return 1 if x == 'Yes' else 0

# Function to preprocess input data and make a prediction
def creditRisk_prediction(data):
    rf = load_model()
    if rf is None:
        return "Model loading failed."

    # Create a DataFrame with the correct column order
    data_prep = pd.DataFrame([data], columns=['person_default', 'home_ownership', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'])

    try:
        prediction = rf.predict_proba(data_prep)[:, 1]
        class_name = "Default" if prediction <= 0.85 else "Non-Default"
        return class_name
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction failed."

# Streamlit app
st.header("Credit Default Prediction App")
st.subheader("Input your data")

# Create input fields for each required feature
person_default_input = st.selectbox("Previous Default", ['No', 'Yes'])
home_ownership_input = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_income = st.number_input("Person Income", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=20.0, format="%.2f")
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, format="%.2f")

# Prepare the data input in the required order
data_input = {
    'person_default': person_default(person_default_input),
    'home_ownership': home_ownership(home_ownership_input),
    'person_income': person_income,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income
}



if data_input is not None:
    if st.button("Analyse"):
        result = creditRisk_prediction(data_input)
        st.subheader("Result:")
        st.info("The result is " + result + ".")