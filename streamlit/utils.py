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

# Streamlit
import streamlit as st

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
    
    
def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    cleanup_nums = {
        "mother_alive": {1: "Yes", 2: "No"},
        "father_alive": {1: "Yes", 2: "No"},
        "marital_status": {
                1: "Married",
                2: "Single",
                3: "Divorced",
                4: "Widowed",
            },
        "parents_level_ed":  {
            1: "No education",
            2: "Religious", 
            3: "Primary",
            4: "Middle School",
            5: "High School",
            6: "Higher Education",
            7: "Professional Training",
        },
        "type_housing": {
            1: "Clay house",
            2: "Permanent house",
            3: "Dry stone",
            4: "Modern/Concrete",
            5: "Other",
        },
        "mobile_phones": {1: "Yes", 2: "No"},
        "individual_water_net": {1: "Yes", 2: "No"},
        "electrical_net_co": {1: "Yes", 2: "No"},
    }
    
    df["parents_age"] = df["parents_age"].fillna(0.0).astype(int)
    df["number_of_person_in_hh"] = df["parents_age"].fillna(0.0).astype(int)
    df["predictions"] = df["predictions"].fillna(0.0).astype(int)

    df = df.drop(columns=["work_activity"])
    df = df.drop(columns=["marital_status"])

    return df.replace(cleanup_nums)


def convert_df(df: pd.DataFrame) -> str:
    return df.to_csv().encode('utf-8')


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

def generate_pdf(figure_list: list) -> io:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in figure_list:
            pdf.savefig(fig)
            plt.close(fig)
    return buffer
