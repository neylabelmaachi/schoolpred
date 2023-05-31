import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Utils
import io
import os
from joblib import load

# Model
from catboost import CatBoostClassifier

# Streamlit
import streamlit as st

dir_path = os.path.dirname(__file__)
model_path = os.path.join(dir_path, '../model/cat_pipeline_2.joblib')

cat_pipeline_model = load(model_path)

# cat_pipeline_model = load('model/cat_pipeline_2.joblib')

st.set_page_config(page_title="Input Manually", page_icon="✍️")


def get_recommendation(data):
        recommendations = []
        
        # Family issues
        # Single household
        if (data['mother_alive'] == 1 or data['father_alive'] == 1):
            if data['marital_status'] != 1:
                recommendations.append("Community program: https://amisdesecoles.org/communaute-solidaire/")
        # Learning issues
        if (data['parents_level_ed'] < 4 or data['average_math_score'] == 'Fail') and data['number_of_person_in_hh'] > 5:
            recommendations.append("Schooling program for unemployed and non-educated mothers: https://www.fondationzakoura.org/alphabetisation.html")
            recommendations.append("Digital schooling program / tablet provider: https://www.fondationzakoura.org/projet_ecole_numerique.html")
        
        return recommendations
    # Inputing individual student data
with st.form(key='params_for_api'):

    mother_alive = st.selectbox("Is the mother alive?", ["Yes", "No"])
    father_alive = st.selectbox("Is the father alive?", ["Yes", "No"])
    parents_age = st.number_input("Age of parent in charge", min_value=15, max_value=80, value=25)
    marital_status = st.selectbox("Marital status", ["Married", "Single","Divorced", "Widowed"])
    parents_level_ed = st.selectbox("Parents' level of education", ["No education", "Religious", "Primary", "Middle School", "High School","Higher Education","Professional Training"])
    work_activity = st.selectbox("Work activity", ["Full Time", "Part Time", "Unemployed"])
    number_of_person_in_hh = st.number_input("Number of people in the household", min_value=1, max_value=25)
    type_housing = st.selectbox('Type of housing', ["Clay house","Permanent house","Dry stone","Modern/Concrete","Other"])
    electrical_net_co = st.selectbox('Does the household have access to electrical network connection?', ["Yes", "No"])
    mobile_phones = st.selectbox("Does the household have one or multiple Mobile phones?", ["Yes", "No"])
    individual_water_net = st.selectbox("Does the household have a personal water connection?", ["Yes", "No"])
    average_math_score = st.selectbox("Average math score", ["Pass", "Fail"])


    # classifying lables according to our data & features
    input_dict = {
        "Yes": 1.0,
        "No": 2.0,
        "Married": 1.0,
        "Single": 2.0,
        "Divorced": 3.0,
        "Widowed": 4.0,
        "No education": 1.0,
        "Religious": 2.0,
        "Primary": 3.0,
        "Middle School": 4.0,
        "High School": 5.0,
        "Higher Education": 6.0,
        "Professional Training": 7.0,
        "Full Time": 1.0,
        "Part Time": 2.0,
        "Unemployed": 3.0,
        "Clay house": 1.0,
        "Permanent house": 2.0,
        "Dry stone": 3.0,
        "Modern/Concrete": 4.0,
        "Other": 5.0,
        "Pass": 2.00,
        "Fail": 1.00,
    }

    # making dataframe dict
    data = {
        "mother_alive": input_dict[mother_alive],
        "father_alive": input_dict[father_alive],
        "parents_age": parents_age,
        "marital_status": input_dict[marital_status],
        "parents_level_ed": input_dict[parents_level_ed],
        "work_activity": input_dict[work_activity],
        "number_of_person_in_hh": number_of_person_in_hh,
        "type_housing": input_dict[type_housing],
        "mobile_phones": input_dict[mobile_phones],
        "individual_water_net": input_dict[individual_water_net],
        "electrical_net_co": input_dict[electrical_net_co],
        "average_math_score": input_dict[average_math_score]
    }


    # creating dataframe
    df = pd.DataFrame(data, index=[0])

    # creating predition button
    submit_button = st.form_submit_button(label='Predict')

    # making model prediction
    if submit_button:
        prediction = cat_pipeline_model.predict(df)[0]
    
        # 1 - Dropout, 0 - enrolled
        # st.warning('This is a warning')
        # st.write(f'The predicted class is {prediction}')
        st.write(prediction)
        
        if prediction == 1:
            st.error('The student is predicted to drop out.')
            recommendations = get_recommendation(data)
            if recommendations:
                st.write("Based on the provided information, the following recommendations are available:")
                for recommendation in recommendations:
                    st.markdown(recommendation)
            else:
                st.write("No recommendations available for the given inputs.")
        else:
            st.success('The student will graduate.')

