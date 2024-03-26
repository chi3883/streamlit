#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:16:57 2023

@author: macos
"""

import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from gettext import NullTranslations, translation
import pandas as pd
import numpy as np
import json
import requests
import altair as alt
import re
import pydeck as pdk
import base64
import datetime as dt
from dateutil.relativedelta import relativedelta 
from statsmodels.tsa.statespace.sarimax import SARIMAX




# to add days or years







st.set_page_config(page_title = "Belib: Paris EV Charging Network", page_icon=":battery:", layout="wide")

#LOAD DATA
path = "https://www.data.gouv.fr/fr/datasets/r/d7326edf-9943-4c41-803a-739008e08434"
df = pd.read_csv(path, delimiter=";", low_memory=False)
    
###SLIDEBAR

image = Image.open('image.PNG')

# Réduire la taille de l'image
new_size = (int(image.width ), int(image.height ))
image_resized = image.resize(new_size)

# Créer un masque circulaire
mask = Image.new("L", new_size, 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0) + new_size, fill=300)

# Appliquer le masque à l'image redimensionnée
image_circle = Image.new("RGBA", new_size)
image_circle.paste(image_resized, mask=mask)

st.sidebar.image(image_circle, use_column_width=True)

st.sidebar.header('NGUYEN Huong-Chi')

st.sidebar.info('**:female_superhero: Crazy student information :** Business Developer at TotalEnergies, Student in Master of Science Data Management-Co-signed by Paris School of Business and Efrei Paris')

icon_size = 50

st.sidebar.markdown('## Social Media')

st.sidebar.markdown('''
[![LinkedIn](https://img.shields.io/badge/linkedin-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/huong-chi-nguyen/)
[![GitHub](https://img.shields.io/badge/github-black?style=for-the-badge&logo=github)](https://github.com/chi3883/)
''')

dataset_name = "Belib' - Points de recharge pour véhicules électriques"
dataset_description = "The Belib EV Charging Points dataset is related to user behavior and characteristics of electric vehicle charging stations located in Paris, France."
page = st.sidebar.radio("**Select mode**", ["Belib' location","Analysis", "Predict"])

st.sidebar.title("Data information")
st.sidebar.markdown("**:zap: Dataset :** " + dataset_name)
st.sidebar.markdown("**:zap: About Dataset :** " + dataset_description)
st.sidebar.write("Source: https://www.data.gouv.fr/fr/datasets/belib-points-de-recharge-pour-vehicules-electriques-donnees-statiques/")




if page == "Belib' location":
    st.title(":zap: Belib' EV Charging Network")
    st.write('Belib’ is the Parisian public network of charging stations for electric vehicles, operated by Total Marketing France (TMF). It has been deployed since March 2021')
    
    data = pd.read_json("map.json")

    # Drop any rows without location coordinates as they cannot be mapped
    data.dropna(inplace=True)

    # Convertir les données en DataFrame pandas
    df = pd.DataFrame(data)

    # Extract longitude and latitude from the "coordonneesxy" column
    data['lon'] = data['coordonneesxy'].apply(lambda x: x['lon'])
    data['lat'] = data['coordonneesxy'].apply(lambda x: x['lat'])
    data['date_maj'] = pd.to_datetime(data['date_maj'])
    data['date_mise_en_service'] = pd.to_datetime(data['date_mise_en_service'])

    # Load icon image and convert to base64
    filepath = "/Users/macos/icon2.png"
    binary_fc = open(filepath, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    ext = filepath.split('.')[-1]
    dataurl = f'data:image/{ext};base64,{base64_utf8_str}'

    # Create a list of icon data dictionaries for each row, considering the filtered data
    icon_data = [{
        "url": dataurl,
        "width": 242,
        "height": 242,
        "anchorY": 242,
    } for _ in range(len(data))]

    # Assign the icon data list to the 'icon_data' column
    data['icon_data'] = icon_data

    st.title(":world_map: Map of Belib': Paris EV Charging Network")
    st.subheader("Density of Belib' Charging point in Paris")
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=48.8592,
            longitude=2.3470,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=data,
                get_position='[lon, lat]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                get_fill_color=[180, 0, 200, 140],
            )
        ]
    ))

    st.subheader("Map of Belib': where to recharge in Paris?")
    st.write(':round_pushpin: Find on the map the different charging stations for your electric vehicles')

    # Liste des types de prises disponibles
    prise_types = ['prise_type_3', 'prise_type_chademo', 'prise_type_2', 'prise_type_ef', 'prise_type_combo_ccs']

    # Sélection des types de prises avec un filtre multiselect
    selected_prise_types = st.multiselect('Select Prise Types', prise_types)

    if not selected_prise_types:
        # Afficher toutes les données si aucun type de prise n'est sélectionné
        filtered_data = data
    else:
        # Filtrer les données en fonction des types de prises sélectionnés
        filtered_data = data[data[selected_prise_types].eq("True").any(axis=1)]

    # Recréer la liste icon_data pour les données filtrées
    icon_data_filtered = [{
        "url": dataurl,
        "width": 242,
        "height": 242,
        "anchorY": 242,
    } for _ in range(len(filtered_data))]

    # Assigner la liste icon_data_filtered à la colonne 'icon_data' des données filtrées
    filtered_data['icon_data'] = icon_data_filtered

    # Créer la couche IconLayer avec les données filtrées
    icon_layer = pdk.Layer(
        type="IconLayer",
        data=filtered_data,
        get_icon="icon_data",
        get_size=4,
        size_scale=5,
        get_position=["lon", "lat"],
        pickable=True,
        tooltip={"text": "{nom_station}, {adresse_station}"}
    )

    # Créer la carte avec les données filtrées
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=48.8592,
            longitude=2.3470,
            zoom=11,
            pitch=50,
        ),
        layers=[icon_layer]
    ))

    
    
elif page=="Analysis":
     # Affichage des résultats dans Streamlit
    st.title(":bar_chart: Belib' : Charging point analysis")
    
    
    # Calcul du nombre d'id_station_local distincts
    nombre_id_station_local = df['id_station_local'].nunique()

# Calcul du nombre de statut_pdc "En service"
    nombre_statut_en_service = df[df['statut_pdc'] == 'En service']['id_station_local'].nunique()


# Calcul du nombre d'id_pdc_local distincts
    nombre_id_pdc_local = df['id_pdc_local'].nunique()
    
    nb_moy = round(nombre_id_pdc_local/nombre_id_station_local)

# Affichage des résultats dans des colonnes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(" :round_pushpin: **Number of charging station**", str(nombre_id_station_local))
    with col2:
        st.metric(":white_check_mark: **Charging stations in service**", str(nombre_statut_en_service))
    with col3:
        st.metric(":battery: **Number of charging points**", str(nombre_id_pdc_local))
    with col4:
        st.metric(":battery: **Avg of charging points per station**", str(nb_moy))
        
    st.write(' With 419 charging stations, 405 of them in service, and 2028 charging points available, there is an average of 5 charging points per station. This suggests a well-distributed charging infrastructure, providing convenient access to charging for electric vehicle owners.')
    
    st.subheader("1 - Analysis by district")

    
    # Calcul du nombre distinct d'id_station_local par arrondissement
    distinct_id_station_local = df.groupby('arrondissement')['id_station_local'].nunique().reset_index()
    distinct_id_station_local.columns = ['Arrondissement', 'Number of charging stations']
    trier_distinct_id_station_local = distinct_id_station_local.sort_values('Number of charging stations', ascending=False)

# Création du graphique
    chart = alt.Chart(trier_distinct_id_station_local).mark_bar().encode(
    y=alt.Y('Arrondissement:N', sort=alt.EncodingSortField(field='Number of charging stations', order='descending'),
            axis=alt.Axis(title='Arrondissement')),
    x=alt.X('Number of charging stations:Q', axis=alt.Axis(title='Number of charging stations'))
    
).properties(
    title="Number of charging stations by arrondissement",
    width=600,
    height=500
)

    chart = chart.configure_axis(
    labelFontSize=10
)

    st.altair_chart(chart)
    

    # Filtrer les données pour les statuts "En service" ou "En test"
    filtered_data = df[df['statut_pdc'].isin(['En service', 'En test'])]

# Calcul du nombre distinct de id_pdc_local par arrondissement et par statut_pdc
    distinct_pdc_count = filtered_data.groupby(['arrondissement', 'statut_pdc'])['id_pdc_local'].nunique().reset_index()
    

# Création du graphique
    chart2 = alt.Chart(distinct_pdc_count).mark_bar().encode(
        x='arrondissement:N',
        y=alt.Y('id_pdc_local:Q', title='Number of charging points'),
        color='statut_pdc:N',
        tooltip=['arrondissement', 'statut_pdc', 'id_pdc_local']
).properties(
        title='Number of charging points by Arrondissement and Statut',
        width=600,
        height=500
    )
    
    df['statut_pdc'] = df['statut_pdc'].str.capitalize()

    fig = px.pie(
        df,
        values=df['statut_pdc'].value_counts().values,
        names=df['statut_pdc'].value_counts().index,
        title="statut_pdc",
        hole=0.6  # Specifies the size of the hole at the center to create a donut chart
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Affichage des graphiques côte à côte
    col1, col2 = st.columns([3, 2])
    col1.altair_chart(chart2)
    col2.plotly_chart(fig)
   
    
    st.write(':first_place_medal: Boroughs 15 and 16 have the most charging stations due to several factors:')
    st.write('**- Population density:** Paris arrondissements 15 and 16 have a relatively high population compared to other arrondissements. A denser population can lead to greater demand for electric car charging stations.')
    st.write("**- Municipal policies:** The municipality of Paris have specific policies or incentives in place to encourage the installation of charging stations in these arrondissements, as part of its initiatives to promote electric mobility and reduce gas emissions.") 
    
   

    
    st.subheader("2 - Analysis by station")
    # Filtrer les données pour le statut "En service"
    filtered_data = df[df['statut_pdc'] == 'En service']
    
    

# Convertir la colonne 'date_mis_en_service' en format de date
    # Convertir la colonne 'date_mise_en_service' en format de date
    filtered_data['date_mise_en_service'] = pd.to_datetime(filtered_data['date_mise_en_service'], format='%Y-%m-%d').dt.date


# Calculer le nombre distinct de id_pdc_local par date
    distinct_pdc_count = filtered_data.groupby('date_mise_en_service')['id_pdc_local'].nunique().reset_index()

# Calculer l'évolution cumulative
    distinct_pdc_count['cumulative_count'] = distinct_pdc_count['id_pdc_local'].cumsum()

# Création du graphique en courbe
    chart = alt.Chart(distinct_pdc_count).mark_line().encode(
        x=alt.X('date_mise_en_service:T',title='Date of commissioning'),
        y=alt.Y('cumulative_count:Q',title='Number of charging points'),
        tooltip=['date_mise_en_service', 'cumulative_count']
).properties(
        title='Cumulative Evolution of Charging points Count (In service)',
        width=700,
        height=400
)
    
    
    
    # Ajouter le filtre de date
    min_date = filtered_data['date_mise_en_service'].min()

    max_date = filtered_data['date_mise_en_service'].max()
    
    selected_dates = st.slider('Select Date Range', min_date, max_date, (min_date, max_date))

    
    # Convertir les valeurs de selected_dates en objets datetime
    start_date = selected_dates[0]
    end_date = selected_dates[1]
    
    # Filter the data based on the selected date range
    filtered_distinct_pdc_count = distinct_pdc_count[
        (distinct_pdc_count['date_mise_en_service'] >= start_date) &
        (distinct_pdc_count['date_mise_en_service'] <= end_date)
]
    
    # Create the chart
    chart = alt.Chart(filtered_distinct_pdc_count).mark_line().encode(
        x=alt.X('date_mise_en_service:T', title='Date of commissioning'),
        y=alt.Y('cumulative_count:Q', title='Number of charging points'),
        tooltip=['date_mise_en_service', 'cumulative_count']
).properties(
        title='Cumulative Evolution of Distinct charging points (In service)',
        width=700,
        height=400
)
    
    st.altair_chart(chart)
    
     # Read the second Excel file into a DataFrame
    other_data = pd.read_excel('utilise.xlsx')

# Perform the merge based on a common column
    merged_data = pd.merge(df, other_data, on='nom_station')
    
    
    
    chart = alt.Chart(merged_data).mark_circle().encode(
    x='Consumption_MWh',
    y='Occupation_rate',
    
).properties(
        title='Correlation between Consumption (MWh) and Occupation rate ')
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
    
    
elif page == "Predict":
    # Read the second Excel file into a DataFrame
    other_data = pd.read_excel('utilise.xlsx')
    st.subheader(':chart_with_downwards_trend: KPI Data')
    st.write(other_data)

    # Read the data from the merged DataFrame
    data = other_data[['Commissioning_date', 'Consumption_MWh']].copy()
    data.dropna(subset=['Commissioning_date', 'Consumption_MWh'], inplace=True)
    data['Consumption_MWh'] = data['Consumption_MWh'].round(2)

    # Convert the 'Commissioning_date' column to datetime
    data['Commissioning_date'] = pd.to_datetime(data['Commissioning_date'], format='%d/%m/%Y')
    data = data[(data['Commissioning_date'].dt.year == 2021) | ((data['Commissioning_date'].dt.year == 2022) & (data['Commissioning_date'].dt.month <= 1))]

    # Sort the data by date
    data.sort_values('Commissioning_date', inplace=True)

    # Remove NaN, infinity, and large values from the data
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Create a 12-month moving average column
    data['AVG12'] = data['Consumption_MWh'].rolling(12).mean()

    # Plot the data and moving average using Plotly Express
    import plotly.express as px
    fig = px.line(data, x="Commissioning_date", y=["Consumption_MWh", "AVG12"])

    # Update the figure size
    fig.update_layout(width=900, height=500)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Extract month and year from dates
    data['Month'] = [i.month for i in data['Commissioning_date']]
    data['Year'] = [i.year for i in data['Commissioning_date']]

    # Create a sequence of numbers
    data['Series'] = np.arange(1, len(data) + 1)

    # Drop unnecessary columns and re-arrange
    data.drop(['AVG12'], axis=1, inplace=True)
    data = data[['Series', 'Year', 'Month', 'Consumption_MWh', 'Commissioning_date']]

    st.subheader(":sparkles: Autoregressive Moving Average (ARMA)")
    st.write('The term “autoregressive” in ARMA means that the model uses past values to predict future ones')

    train = data[data['Commissioning_date'] < '12/01/2021']
    test = data[data['Commissioning_date'] >= '12/01/2021']

    # Plot the train and test data
    fig_train_test = px.line(data_frame=data, x="Commissioning_date", y="Consumption_MWh", title="Train and Test Data")
    fig_train_test.add_trace(px.line(test, x="Commissioning_date", y=["Consumption_MWh"]).data[0])

    # Update the line colors for train and test
    fig_train_test.data[-1].line.color = 'red'  # Test data color

    # Update the figure size
    fig_train_test.update_layout(width=900, height=500)

    # Make predictions for the next 3 months
    prediction_start_date = data['Commissioning_date'].max()
    prediction_end_date = prediction_start_date + pd.DateOffset(months=3)
    prediction_dates = pd.date_range(prediction_start_date, prediction_end_date)

    # Define the ARIMA model
    y_train = train['Consumption_MWh']
    ARIMAModel = SARIMAX(y_train, order=(2, 2, 2))
    ARIMAModel = ARIMAModel.fit()
    y_pred = ARIMAModel.get_forecast(len(prediction_dates))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARIMAModel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = prediction_dates
    y_pred_out = y_pred_df["Predictions"]

    # Plot the predictions
    fig_train_test.add_trace(go.Scatter(x=y_pred_out.index, y=y_pred_out, name='ARMA Predictions', line=dict(color='green')))

    # Update the figure layout
    fig_train_test.update_layout(showlegend=True)

    # Display the plot in Streamlit
    st.plotly_chart(fig_train_test)

   

    
   

