import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pydeck as pdk


st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.title('Primary School Dropout Predictor')
st.write("""This tool has been built in order to prevent primary school dropout in high-risk rural regions of Morocco.
        Our prediction model has been trained on the following research dataset:
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
