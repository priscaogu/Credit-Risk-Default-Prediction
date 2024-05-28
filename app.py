import pickle
import pandas as pd
import streamlit as st

# Function to load the model
def load_model():
    with open("cr_model.pkt", "rb") as file:
        rf = pickle.load(file)
    return rf

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
def previous_default(x):
    return 0 if x == 'No' else 1

# Function to preprocess input data and make a prediction
def creditRisk_prediction(data):
    rf = load_model()
    # Create a DataFrame with the correct column order
    data_prep = pd.DataFrame([data], columns=['previous_default', 'home_ownership', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'])
    prediction = rf.predict_proba(data_prep)[:, 1]
    class_name = "Default" if prediction <= 0.85 else "Non-Default"
    return class_name

# Streamlit app
st.header("Credit Default Prediction App")
st.subheader("Input your data")

# Create input fields for each required feature
person_default_input = st.selectbox("Previous Default", ['No', 'Yes'])
home_ownership_input = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_income = st.number_input("Person Income", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, format="%.2f")
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, format="%.2f")

# Prepare the data input in the required order
data_input = {
    'previous_default': previous_default(previous_default_input),
    'home_ownership': home_ownership(home_ownership_input),
    'person_income': person_income,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income
}

if text_input is not None:
    if st.button("Analyse"):
        result = creditRisk_prediction(data_input)
        st.subheader("Result:")
        st.info("The result is " + result + ".")