import streamlit as st
import pandas as pd
import pydeck as pdk
import base64

# Function to load data (with caching to speed up app reloading)
@st.cache
def load_data(url):
    df = pd.read_csv(url, delimiter=";", low_memory=False)
    return df

# Function to encode icon image to base64 (for displaying custom icons on the map)
@st.cache
def encode_icon_to_base64(icon_path):
    with open(icon_path, "rb") as icon_file:
        base64_icon = base64.b64encode(icon_file.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_icon}"

# Main function to display the map
def display_charging_station_map(data, icon_url):
    # Define the initial view state of the map
    view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11, pitch=0)

    # Define the icon layer
    icon_layer = pdk.Layer(
        "IconLayer",
        data,
        get_icon="icon_data",
        get_size=4,
        size_scale=15,
        get_position=["lon", "lat"],
        pickable=True,
        tooltip={"text": "{nom_station}, {adresse_station}"}
    )

    # Render the map with the icon layer
    st.pydeck_chart(pdk.Deck(layers=[icon_layer], initial_view_state=view_state))

# Load the data
DATA_URL = "https://www.data.gouv.fr/fr/datasets/r/d7326edf-9943-4c41-803a-739008e08434"
df = load_data(DATA_URL)

# Assuming your dataset already has 'lat' and 'lon' columns. If not, adjust accordingly.
# For demonstration, let's add some mock coordinates if your dataset doesn't have them.
# df['lat'] = [48.8566 for _ in range(len(df))]
# df['lon'] = [2.3522 for _ in range(len(df))]

# Prepare the icon data, including the URL to the base64-encoded icon
ICON_PATH = "path/to/your/icon.png"
icon_url = encode_icon_to_base64(ICON_PATH)
df['icon_data'] = [{'url': icon_url, 'width': 128, 'height': 128, 'anchorY': 128} for _ in range(len(df))]

# Display the map
st.title("Belib' Charging Stations in Paris")
display_charging_station_map(df, icon_url)
