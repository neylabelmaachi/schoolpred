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
uploaded_file = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=True)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, sep=';')
        X = data.drop(columns=["age_dropout"])
        y = data["age_dropout"]
        predictions = cat_pipeline_model.predict(X)
        st.write("Students at risk of school dropout are the following:")
        st.write(predictions)

    except pd.errors.ParserError:
        st.error("Error: Invalid CSV file. Please upload a valid CSV file.")

# Inputing individual student data
with st.form(key='params_for_api'):

    mother_alive = st.selectbox("Is the mother alive?", ["Yes", "No"])
    father_alive = st.selectbox("Is the father alive?", ["Yes", "No"])
    parents_age = st.number_input("Age of parent in charge", min_value=0, max_value=150)
    marital_status = st.selectbox("Marital status", ["Married", "Single","Divorced", "Widowed"])
    parents_level_ed = st.selectbox("Parents' level of education", ["No education", "Religious", "Primary", "Middle School", "High School","Higher Education","Professional Training"])
    number_of_person_in_hh = st.number_input("Number of people in the household", min_value=0)
    mobile_phones = st.selectbox("Does the household have one or multiple Mobile phones?", ["Yes", "No"])
    individual_water_net = st.selectbox("Does the household have a personal water connection?", ["Yes", "No"])
    work_activity = st.selectbox("Work activity", ["Full Time", "Part Time", "Unemployed"])
    average_math_score = st.number_input("Average math score", min_value=0, max_value=100)

    submit_button = st.form_submit_button("Submit")

# Access the submitted values after submission
if submit_button:
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
