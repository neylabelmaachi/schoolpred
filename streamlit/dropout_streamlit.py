import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from catboost import CatBoostClassifier
import joblib
import json
from joblib import load


st.title('Primary School Dropout Predictor')
st.write("Please upload a CSV file with classroom data to predict the probability of student dropout.")

cat_pipeline_model = load('cat_pipeline_2.joblib')

# Uploading a CSV file
uploaded_files = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=False)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        try:
            data = pd.read_csv(uploaded_file)
            print(data.columns)
            X = data.drop(columns=["age_dropout"])
            predictions = cat_pipeline_model.predict(X)
            st.write("Students at risk of school dropout are the following:")
            st.write(predictions)

        except pd.errors.ParserError:
            st.error("Error: Invalid CSV file. Please upload a valid CSV file.")

# Inputing individual student data
with st.form(key='params_for_api'):

    mother_alive = st.selectbox("Is the mother alive?", ["Yes", "No"])
    father_alive = st.selectbox("Is the father alive?", ["Yes", "No"])
    parents_age = st.number_input("Age of parent in charge", min_value=0, max_value=99)
    marital_status = st.selectbox("Marital status", ["Married", "Single","Divorced", "Widowed"])
    parents_level_ed = st.selectbox("Parents' level of education", ["No education", "Religious", "Primary", "Middle School", "High School","Higher Education","Professional Training"])
    work_activity = st.selectbox("Work activity", ["Full Time", "Part Time", "Unemployed"])
    number_of_person_in_hh = st.number_input("Number of people in the household", min_value=0, max_value=25)
    type_housing = st.selectbox('Type of housing', ["Clay house","Permanent house","Dry stone","Modern/Concrete","Other"])
    electrical_net_co = st.selectbox('Does the household have access to electrical network connection?', ["Yes", "No"])
    mobile_phones = st.selectbox("Does the household have one or multiple Mobile phones?", ["Yes", "No"])
    individual_water_net = st.selectbox("Does the household have a personal water connection?", ["Yes", "No"])
    average_math_score = st.selectbox("Average math score", ["Pass", "Fail"])


    # Make predictions on form submission
    if st.form_submit_button("Submit"):

        input_data = pd.DataFrame({
            'mother_alive': [mother_alive],
            'father_alive': [father_alive],
            'parents_age': [parents_age],
            'marital_status': [marital_status],
            'parents_level_ed': [parents_level_ed],
            "type_housing" : [type_housing],
            "electrical_net_co": [electrical_net_co],
            'number_of_person_in_hh': [number_of_person_in_hh],
            'mobile_phones': [mobile_phones],
            'individual_water_net': [individual_water_net],
            'work_activity': [work_activity],
            'average_math_score': [average_math_score]
        })

        # Make predictions using the loaded model
        predictions = cat_pipeline_model.predict(input_data)

        # Display the predictions
        st.write("Predicted probability of student dropout:")
        st.write(predictions)

# Access the submitted values after submission
if st.form_submit_button is not None:
    st.write("Submitted values:")
    st.write("Mother alive:", mother_alive)
    st.write("Father alive:", father_alive)
    st.write("Parents' age:", parents_age)
    st.write("Marital status:", marital_status)
    st.write("Parents' level of education:", parents_level_ed)
    st.write("Number of people in household:", number_of_person_in_hh)
    st.write("Access to mobile phones:", mobile_phones)
    st.write("Access to individual water connection:", individual_water_net)
    st.write("Work activity:", work_activity)
    st.write("Average math score:", average_math_score)
    st.write("Type of housing", type_housing)
    st.write("Electrical network connection", electrical_net_co)
