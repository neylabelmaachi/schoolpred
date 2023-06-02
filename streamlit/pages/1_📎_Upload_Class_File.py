import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Utils
import io
from joblib import load
from utils import *
import os

# Model
from catboost import CatBoostClassifier

# Streamlit
import streamlit as st

st.set_page_config(page_title="Upload your file", page_icon="📎")

st.title('Primary School Dropout Predictor')
st.write("Please upload a CSV file with classroom data to predict the probability of student dropout.")

dir_path = os.path.dirname(__file__)
model_path = os.path.join(dir_path, '../../model/cat_pipeline_2.joblib')

cat_pipeline_model = load(model_path)

data = None
figure_list = []

try:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    data = pd.read_csv(uploaded_file)
except:
    st.write("CSV upload failed")
    
student_tab, summary_tab = st.tabs(["Students", "Summary"])

if data is not None:
    predictions = cat_pipeline_model.predict(data)
    predicted_proba = cat_pipeline_model.predict_proba(data)
    data["predictions"] = predictions
    data["dropout_proba"] = predicted_proba[:, 1]
    # Family Issue
    data['recommendations'] = data.apply(lambda row: get_recommendation(row), axis=1)


with student_tab:
    if data is not None:
        st.header("Students")
        # Encode Data frame
        encoded = encode_df(data)

        filtered_df = filter_dataframe(encoded)
        st.dataframe(filtered_df.style.applymap(style_predictions, subset=["predictions"]))
        
        st.download_button(
            label="Download data as CSV",
            data=convert_df_to_csv(filtered_df),
            file_name='students.csv',
            mime='text/csv'
        )

        
with summary_tab:
    if data is not None:
        st.header("Summary")

        # Dropout probability percentage
        fig_dropout = plt.figure(figsize=(10,6))

        plt.title("Possible Dropout")
        plt.pie(data['predictions'].value_counts(),  labels = ['Graduate', 'Dropout'], explode = (0.1, 0.0), autopct='%1.2f%%', shadow = True)
        plt.legend()
        figure_list.append(fig_dropout)
        st.pyplot(fig_dropout)
        
        # Family Profile
        st.subheader('Family Profile')
        fig_family = plt.figure()
        plt.xticks(rotation=45)

        sns.countplot(data=data, x="number_of_person_in_hh", hue="predictions")
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])
        plt.xlabel('Number of Household members')
        plt.ylabel('Number of Students')

        figure_list.append(fig_family)
        st.pyplot(fig_family)
        
        fig_father = plt.figure()
        sns.countplot(data=data, x="father_alive", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes', 'No'])
        plt.xlabel('Father Alive')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        figure_list.append(fig_father)
        st.pyplot(fig_father)
        
        fig_mother = plt.figure()
        sns.countplot(data=data, x="mother_alive", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes', 'No'])
        plt.xlabel('Mother Alive')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        figure_list.append(fig_mother)
        st.pyplot(fig_mother)
        
        fig_marital = plt.figure()
        
        mar_status = {
            1: "married",
            2: "single",
            3: "divorced",
            4: "widowed"
        }
        
        data["marital_status_encoded"] = data["marital_status"].map(mar_status)

        sns.countplot(data=data, x="marital_status_encoded", hue="predictions")
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])
        plt.xlabel("Parent's Marital Status")
        plt.ylabel('Number of Students')
        
        figure_list.append(fig_marital)
        st.pyplot(fig_marital)

        # Academic performance
        st.subheader("Student's Academic Performance")
        fig_acad = plt.figure()

        sns.boxplot(data=data, x="predictions", y="average_math_score", order=[0,1])
        plt.xticks(ticks=[0,1], labels=['Graduate', 'Dropout'], rotation=24)
        plt.ylabel('Math Scores')
        
        figure_list.append(fig_acad)
        st.pyplot(fig_acad)
        
        # Economic status
        st.subheader("Family's Socio Economic Status")

        work_cat = {
            1: "permanent", 
            2: "permanent", 
            3: "part_time", 
            9: "unemployed", 
            4: "unemployed",
            5: "unemployed",
            7: "unemployed",
            8: "unemployed"
        }
        data["work_activity_encoded"] = data["work_activity"].map(work_cat)
        
        fig_work_status = plt.figure()
        sns.countplot(data=data, x="work_activity_encoded", hue="predictions")
        plt.xlabel('Work Status')
        plt.ylabel('Number of Students')

        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        figure_list.append(fig_work_status)
        st.pyplot(fig_work_status)
        
        fig_water = plt.figure()
        sns.countplot(data=data, x="individual_water_net", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Have Water at home')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        figure_list.append(fig_water)
        st.pyplot(fig_water)
        
        fig_elec = plt.figure()
        sns.countplot(data=data, x="electrical_net_co", hue='predictions', hue_order=[0, 1])
        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Have Electricity at home')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        figure_list.append(fig_elec)
        st.pyplot(fig_elec)
        
        fig_mob_phone = plt.figure()
        sns.countplot(data=data, x="mobile_phones", hue='predictions', hue_order=[0, 1])
        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Family have atleast 1 mobile phone at home')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])
        
        figure_list.append(fig_mob_phone)
        st.pyplot(fig_mob_phone)
        
        fig_housing = plt.figure()
        sns.countplot(data=data, x="type_housing", hue='predictions', hue_order=[0, 1])
        plt.xticks(ticks=[0,1,2,3,4], labels=['Adobe','Permanent', 'Dry Stone', 'Modern', 'Other'])
        plt.xlabel('Type of house')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Graduate', 'Dropout'])

        
        figure_list.append(fig_housing)
        st.pyplot(fig_housing)        
        
        # Geography
        
        # Generate PDF buffer
        pdf_buffer = generate_pdf(figure_list)
        
        pdf_buffer.seek(0)
        st.download_button(label="Export Report", data=pdf_buffer, file_name="summary.pdf", mime='application/octet-stream')

