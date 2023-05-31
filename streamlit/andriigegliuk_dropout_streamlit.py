
import os
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pydeck as pdk



dir_path = os.path.dirname(__file__)
model_path = os.path.join(dir_path, '../model/cat_pipeline_2.joblib')

cat_pipeline_model = load(model_path)


# cat_pipeline_model = load('../model/cat_pipeline_2.joblib')


page = st.sidebar.selectbox(
    'Select Your Analysis Task:',
    ['🏠 Home', '📎 Upload classroom file', '✍️ Enter Individual Student Data']
)

if page == '🏠 Home':



    st.title('Primary School Dropout Predictor')
    st.write("This tool has been built in order to prevent primary school dropout in high-risk rural regions of Morocco.")
    st.write("""Our prediction model has been trained on the following research dataset:
            Data for Development Initiative. (2019). Morocco CCT Education (Version 1.0)
            [Data set]. Redivis. https://redivis.com/datasets/11xy-bb1z6q7ap?v=1.0""")

    st.title('Identifying and preventing risks')
    st.write("Please upload a CSV file with classroom data or manually input individual student data to predict the probability of school dropout.")



    data = {
    'province': ['Khenifra', 'Taroudant', 'Taourirt', 'Ouarzazate', 'Azilal',
                 'Chtouka Ait Baha', 'Essaouira', 'El Kelaa Des Sraghna', 'Errachidia',
                 'Meknes', 'Chichaoua', 'Nador', 'Tiznit', 'Al Haouz', 'Ifrane',
                 'Jerada', 'El Hajeb'],
    'latitude': [32.935772, 30.470651, 34.413438, 30.920193, 31.870226,
                 30.070334, 31.511828, 32.054330, 31.929089, 33.898413,
                 31.546903, 35.051739, 29.698624, 30.964141, 33.527605,
                 34.310531, 33.698752],
    'longitude': [-5.669650, -8.877922, -2.893825, -6.910923, -6.432248,
                  -9.154578, -9.762090, -7.406760, -4.434081, -5.532158,
                  -8.759546, -2.824651, -9.731281, -8.100704, -5.107408,
                  -2.159979, -5.492859]
    }

    df = pd.DataFrame(data)

    view_state = pdk.ViewState(
        latitude = df['latitude'].mean(),
        longitude = df['longitude'].mean(),
        zoom = 5,
        pitch = 0)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data = df,
        get_position = '[longitude, latitude]',
        get_radius = 20000,
        get_fill_color = [255, 0, 0, 180],
        pickable = True,
        auto_highlight = True
    )


    tooltip = {
        "html": "<b>{province}</b>",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    st.pydeck_chart(pdk.Deck(
        map_style = None,
        initial_view_state = view_state,
        layers = [layer],
        tooltip = tooltip
    ))


elif page == '📎 Upload classroom file':
    # Uploading a CSV file
    uploaded_files = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=False)

    # if uploaded_files is not None:
    #     for uploaded_file in uploaded_files:
    #         try:
    #             data = pd.read_csv(io.BytesIO(uploaded_file), sep=';')
    #             data = data.set_index('hhid')
    #             X = data.drop("age_dropout", axis=1)
    #             y = data['age_dropout']
    #             predictions = cat_pipeline_model.predict(X)
    #             st.write("Students at risk of school dropout are the following:")
    #             st.write(predictions)

    #         except pd.errors.ParserError:
    #             st.error("Error: Invalid CSV file. Please upload a valid CSV file.")

    if uploaded_files is not None:
    # Read the CSV file into a DataFrame
        data = pd.read_csv(uploaded_files)

        # Display the DataFrame
        st.dataframe(data)

        # Predict with the model
        predictions = cat_pipeline_model.predict(data)

        print(predictions)
        st.dataframe(predictions)

elif page == '✍️ Enter Individual Student Data':
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

            if prediction == 0:
                st.success('The student will graduate.')
            else:
                st.error('The student is predicted to drop out.')
