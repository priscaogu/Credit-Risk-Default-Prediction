import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier



def load_model():
    rf_model = pickle.load(open("cr_model.pkt", "rb"))
    return rf_model


# Streamlit app
st.header("Credit Default Prediction App")
st.subheader("Input your data")
#data_input = st.data_input("Enter your data")

# Create input fields for each required feature
person_default_input = st.selectbox("Previous Default", ['No', 'Yes'])
home_ownership_input = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_income = st.number_input("Person Income", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=20.0, format="%.2f")
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, format="%.2f")

# Function to convert home ownership to numerical value
def classify_home(x):
    if x == 'OWN':
        return 0
    elif x == 'MORTGAGE':
        return 1
    elif x == 'RENT':
        return 2
    else:
        return 3


# Function to convert person default to numerical value
def classify_previousdefault(x):
    return 1 if x == 'Yes' else 0


#Ordinal value arrangement 
classify_home = home_ownership(home_ownership_input)
classify_previousdefault = person_default(person_default_input)



# Function to preprocess input data and make a prediction
def creditRisk_prediction(data):
    rf_model = load_model()
    if rf_model is None:
        return "Model loading failed."

    try:
        prediction = rf.predict_proba(data)[:, 1]
        class_name = "Default" if prediction <= 0.85 else "Non-Default"
        return class_name
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction failed."

# Prepare the data input
# Function to prepare data input
data_input = np.array([person_default, home_ownership, person_income, loan_amnt, loan_int_rate, loan_percent_income] , ndmin=2)
    

if data_input is not None:
    if st.button("Analyse"):
        result = creditRisk_prediction(data_input)
        st.subheader("Result:")
        st.info("The result is " + result + ".")