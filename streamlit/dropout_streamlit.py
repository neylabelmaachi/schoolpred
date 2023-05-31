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
from joblib import load

# Model
from catboost import CatBoostClassifier

# Streamlit
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components



st.title('Primary School Dropout Predictor')
st.write("Please upload a CSV file with classroom data to predict the probability of student dropout.")

cat_pipeline_model = load('model/cat_pipeline_2.joblib')
data = None

try:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    data = pd.read_csv(uploaded_file)
except:
    st.write("CSV upload failed")
    
    
tab1, tab2 = st.tabs(["Summary", "Students"])

with tab1:
    st.header("Summary")
    
    def get_recommendation(row):
        recommendations = []
        
        # Family issues
        if row['mother_alive'] == 2 and row['father_alive'] == 2 and row['parents_age'] > 50 and row['marital_status'] == 4:
            recommendations.append("Community program: https://amisdesecoles.org/communaute-solidaire/")
        
        # Learning issues
        if row['parents_level_ed'] < 4 and row['average_math_score'] >= 1.75 and row['number_of_person_in_hh'] > 5:
            recommendations.append("Schooling program for unemployed and non-educated mothers: https://www.fondationzakoura.org/alphabetisation.html")
            recommendations.append("Digital schooling program / tablet provider: https://www.fondationzakoura.org/projet_ecole_numerique.html")
        
        return recommendations
        
        
    if data is not None:
        # Predict with the model 
        predictions = cat_pipeline_model.predict(data)
        predicted_proba = cat_pipeline_model.predict_proba(data)
        # prediction_label = {
        #     0: "Enrolled",
        #     1: "Dropout"
        # }
        data["predictions"] = predictions
        # data["predictions_encoded"] = data["predictions"].map(prediction_label)
        data["enrolled_proba"] = predicted_proba[:, 0]
        data["dropout_proba"] = predicted_proba[:, 1]
        # Family Issue
        data['recommendations'] = data.apply(lambda row: get_recommendation(row), axis=1)
        
        st.dataframe(data)
        #   Single household
        #   1 - yes, 0 - no
        
        #   Parent's age
        
        # Learning Issue
        #   Parent's ed level
        #   Math score

        # Dropout probability percentage
        fig_dropout = plt.figure(figsize=(16, 8))
        plt.title("Possible Dropout")
        plt.pie(data['predictions'].value_counts(),  labels = ['Non-Dropout', 'Dropout'], explode = (0.1, 0.0), autopct='%1.2f%%', shadow = True)
        plt.legend()
        
        st.pyplot(fig_dropout)
        
        # Family Profile
        st.subheader('Family Profile')
        fig_family = plt.figure(figsize=(10, 4))
        plt.xticks(rotation=45)

        sns.countplot(data=data, x="number_of_person_in_hh", hue="predictions")
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])
        plt.xlabel('Number of Household members')
        
        st.pyplot(fig_family)
        
        fig_father = plt.figure(figsize=(10, 4))
        sns.countplot(data=data, x="father_alive", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes', 'No'])
        plt.xlabel('Father Alive')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])

        st.pyplot(fig_father)
        
        fig_mother = plt.figure(figsize=(10, 4))
        sns.countplot(data=data, x="mother_alive", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes', 'No'])
        plt.xlabel('Mother Alive')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])

        st.pyplot(fig_mother)
        
        fig_marital = plt.figure(figsize=(10, 4))
        
        mar_status = {
            1: "married",
            2: "single",
            3: "divorced",
            4: "widowed"
        }
        
        data["marital_status_encoded"] = data["marital_status"].map(mar_status)

        sns.countplot(data=data, x="marital_status_encoded", hue="predictions")
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])
        plt.xlabel("Parent's Marital Status")

        st.pyplot(fig_marital)


        # Academic performance
        st.subheader("Student's Academic Performance")

        fig_acad = plt.figure(figsize=(10, 4))

        # sns.scatterplot(data=data, x="predictions", y="average_math_score", hue="predictions")
        sns.boxplot(data=data, x="predictions", y="average_math_score", hue="predictions")
        
        plt.xticks(ticks=[0,1], labels=['Yes','No'], rotation=24)
        plt.xlabel('Average Math Score')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])
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
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])
        st.write(data["work_activity_encoded"].value_counts())
        st.write(data["work_activity"].value_counts())


        st.pyplot(fig_work_status)
        
        fig_water = plt.figure()
        sns.countplot(data=data, x="individual_water_net", hue='predictions', hue_order=[0, 1])

        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Have Water at home')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])

        st.pyplot(fig_water)
        
        fig_elec = plt.figure()
        sns.countplot(data=data, x="electrical_net_co", hue='predictions', hue_order=[0, 1])
        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Have Electricity at home')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])

        st.pyplot(fig_elec)
        
        fig_mob_phone = plt.figure()
        sns.countplot(data=data, x="mobile_phones", hue='predictions', hue_order=[0, 1])
        plt.xticks(ticks=[0,1], labels=['Yes','No'])
        plt.xlabel('Family have atleast 1 mobile phone at home')
        plt.ylabel('Number of Students')
        
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])
        st.pyplot(fig_mob_phone)
        
        
        fig_housing = plt.figure()
        sns.countplot(data=data, x="type_housing", hue='predictions', hue_order=[0, 1])
        # st.write(data["type_housing"].value_counts())
        plt.xticks(ticks=[0,1,2,3,4], labels=['Adobe','Permanent', 'Dry Stone', 'Modern', 'Other'])
        plt.xlabel('Type of house')
        plt.ylabel('Number of Students')
        plt.legend(title="Predictions", labels = ['Non-Dropout', 'Dropout'])

        st.pyplot(fig_housing)        
        
        # Geography
        
        # Save figures to PDF
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig_dropout)
            pdf.savefig(fig_family)
            pdf.savefig(fig_father)
            pdf.savefig(fig_mother)
            pdf.savefig(fig_marital)
            pdf.savefig(fig_acad)
            pdf.savefig(fig_work_status)
            pdf.savefig(fig_water)
            pdf.savefig(fig_elec)
            pdf.savefig(fig_mob_phone)
            pdf.savefig(fig_water)
            pdf.savefig(fig_housing)

        # Close figures
        plt.close(fig_dropout)
        plt.close(fig_family)
        plt.close(fig_father)
        plt.close(fig_mother)
        plt.close(fig_marital)
        plt.close(fig_acad)
        plt.close(fig_work_status)
        plt.close(fig_water)
        plt.close(fig_elec)
        plt.close(fig_mob_phone)
        plt.close(fig_water)
        plt.close(fig_housing)
        
        pdf_buffer.seek(0)
        st.download_button(label="Export_Report", data=pdf_buffer, file_name="test.pdf", mime='application/octet-stream')

with tab2:
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters")

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]

        return df
    
    def convert_df(df: pd.DataFrame) -> str:
        return df.to_csv().encode('utf-8')

    if data is not None:
        st.header("Students")
        
        filtered_df = filter_dataframe(data)
        st.dataframe(filtered_df)
        
        csv = convert_df(filtered_df)
        
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )
        