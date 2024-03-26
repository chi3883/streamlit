import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import pydeck as pdk
import base64

# Set page config
st.set_page_config(page_title="Belib: Paris EV Charging Network", page_icon=":battery:", layout="wide")

def load_data(url):
    """Load CSV data from a URL."""
    return pd.read_csv(url, delimiter=";", low_memory=False)

@st.cache(allow_output_mutation=True)
def load_image(image_path):
    """Load and process an image."""
    image = Image.open(image_path)
    # Create a circular mask to apply to the image
    mask = Image.new("L", (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.width, image.height), fill=255)
    image_circle = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    image_circle.putalpha(mask)
    return image_circle

@st.cache
def load_icon(icon_path):
    """Load and encode an icon to base64."""
    with open(icon_path, "rb") as icon_file:
        return base64.b64encode(icon_file.read()).decode("utf-8")

def display_location_data(data, icon_data_url):
    """Display the map with location data and icons."""
    # Adjust the icon data for pydeck
    for item in data.itertuples():
        item.icon_data = {
            "url": icon_data_url,
            "width": 242,
            "height": 242,
            "anchorY": 242,
        }
    
    view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=10, bearing=0, pitch=0)
    icon_layer = pdk.Layer(
        type="IconLayer",
        data=data,
        get_icon="icon_data",
        get_size=4,
        size_scale=15,
        get_position=["lon", "lat"],
        pickable=True,
        tooltip={"text": "{nom_station}, {adresse_station}"}
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[icon_layer]))

# Main
def main():
    DATA_URL = "https://www.data.gouv.fr/fr/datasets/r/d7326edf-9943-4c41-803a-739008e08434"
    IMAGE_PATH = "image.png"
    ICON_PATH = "icon.png"

    df = load_data(DATA_URL)
    sidebar_image = load_image(IMAGE_PATH)
    icon_data_url = f"data:image/png;base64,{load_icon(ICON_PATH)}"

    st.sidebar.image(sidebar_image, use_column_width=True)
    st.sidebar.header('NGUYEN Huong-Chi')
    st.sidebar.info('**:female_superhero: Crazy student information :** Business Analyst at TotalEnergies, Student in Master of Science Data Management-Co-signed by Paris School of Business and Efrei Paris')

    st.sidebar.markdown('## Social Media')
    st.sidebar.markdown('''
    [![LinkedIn](https://img.shields.io/badge/linkedin-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/huong-chi-nguyen/)
    [![GitHub](https://img.shields.io/badge/github-black?style=for-the-badge&logo=github)](https://github.com/chi3883/)
    ''')

    st.title(":zap: Belib' EV Charging Network")
    st.write('Belib’ is the Parisian public network of charging stations for electric vehicles, operated by Total Marketing France (TMF). It has been deployed since March 2021.')
    # Assuming `data` is a DataFrame with columns ['lon', 'lat', 'nom_station', 'adresse_station'].
    # Adjust your DataFrame accordingly.
    display_location_data(df, icon_data_url)

if __name__ == "__main__":
    main()
