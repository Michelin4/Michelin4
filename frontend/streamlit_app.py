import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import geocoder
import folium
import pandas as pd
import streamlit.components.v1 as components

# FOLIUM IMPLEMENTATION
class Map:
    def __init__(self, data_point, zoom_start=15):
        self.center = (data_point['LATITUDE'], data_point['LONGITUDE'])
        self.zoom_start = zoom_start

        tiles = 'https://tile.jawg.io/jawg-dark/{z}/{x}/{y}{r}.png?access-token=yKizQwgQd53rog6n0kisnZ6OhqsNUzcMIhPKV7RLevpkLFUEfGrQbgW8SUl9GwKa'
        attr = (
            '<a href="https://jawg.io" title="Tiles Courtesy of Jawg Maps" target="_blank">&copy; <b>Jawg</b>Maps</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        )
        self.map = folium.Map(location=self.center, zoom_start=self.zoom_start, tiles=tiles, attr=attr)

        self.dict_color_poi = {
            'highway': 'darkpurple',
            'tourism': 'orange',
            'shop': 'blue',
            'amenity': 'green',
            'surface': 'purple',
            'leisure': 'pink',
            'building': 'black',
        }

        self.dict_icons_poi = {  # From https://fontawesome.com/v4/icons/
            'highway': 'road',
            'tourism': 'suitcase',
            'shop': 'shopping-bag',
            'amenity': 'bath',
            'surface': 'expand',
            'leisure': 'gift',
            'building': 'building',
        }

    def get_map_html(self):
        """Save map to HTML and return its content."""
        map_path = "map.html"
        self.map.save(map_path)
        with open(map_path, "r") as file:
            return file.read()
        
    def extract_coordinates(prompt: str):
        """Extract coordinates from the user-provided prompt."""
        import re
        pattern = r"[-+]?[0-9]*\.?[0-9]+"
        numbers = re.findall(pattern, prompt)
        if len(numbers) >= 2:
            lat, lon = map(float, numbers[:2])
            return lat, lon
        else:
            raise ValueError("Could not extract valid coordinates from the prompt.")

    def marker(self, point, icon='circle', color='red', popup=False):
        lat, lon = point['LATITUDE'], point['LONGITUDE']
        if popup:
            injury_columns = ['FATAL', 'SERIOUS_INJURY', 'MINOR_INJURY', 'NO_INJURY']
            injury_type = point[injury_columns][point[injury_columns] == 1.0].index[0]

            person_involved = ['VH', 'CYC', 'PED']
            pi_type = point[person_involved][point[person_involved] == 1.0].index[0]

            popup_content = (
                f"Latitude: {point['LATITUDE']}<br>"
                f"Longitude: {point['LONGITUDE']}<br>"
                f"Year: {int(point['YEAR'])}<br>"
                f"Person Involved: {pi_type}<br>"
                f"Injury: {injury_type}<br>"
                f"POI Type: {point['nearest_poi_type']}<br>"
                f"Distance to POI: {point['distance_to_poi']}"
            )

            folium.Marker(
                location=(lat, lon),
                icon=folium.Icon(color=color, prefix='fa', icon=icon),
                popup=folium.Popup(popup_content, min_width=300, max_width=300),
            ).add_to(self.map)
        else:
            folium.Marker(
                location=(lat, lon),
                icon=folium.Icon(color=color, prefix='fa', icon=icon),
            ).add_to(self.map)



# FRONTEND WITH BACKEND INTEGRATION
# Set Streamlit page configuration
st.set_page_config(page_title="Michelin Tires Chatbot")

# Sidebar title
with st.sidebar:
    st.title("Michelin Tires Chatbot")

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Prompt for API key only if not already provided
if not st.session_state.api_key:
    st.sidebar.subheader("Enter OpenAI API Key")
    api_key_input = st.sidebar.text_input("API Key:", type="password")
    if st.sidebar.button("Submit"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.sidebar.success("API Key saved!")
        else:
            st.sidebar.error("Please enter a valid API key.")

# Ensure API key is available before proceeding
if not st.session_state.api_key:
    st.warning("Please provide your OpenAI API key in the sidebar to use the application.")
else:
    # Use the stored API key
    api_key = st.session_state.api_key

    # Load data
    df_crash = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")
    df_acceleration = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/harsh_accel_data.csv")
    df_braking = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/harsh_braking_data.csv")

    # Tool: Calculate Distance
    @tool
    def calculate_distance(lat1: float, lon1: float) -> list[dict]:
        """
        Calculate the distance in miles between a given point (lat1, lon1) and 
        each row's geographic point using the Haversine formula. Add a `distance` column.
        """
        df = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")

        def haversine(row):
            lat2, lon2 = row['LATITUDE'], row['LONGITUDE']
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            r = 3963  # Radius of Earth in miles
            return r * c

        df['distance'] = df.apply(haversine, axis=1)
        df.to_csv("/Users/sevinchnoori/Michelin4/processed_data/temp.csv")
        return df

    # Tool: Get Geographic Coordinates
    @tool
    def get_geographic_coordinates(placename: str) -> tuple:
        """
        Fetch geographic coordinates (latitude and longitude) for a given place name.
        """
        try:
            search = geocoder.geonames(placename, maxRows=1, country='US', adminCode1='CA', adminCode2='037', key='dongim04')
            latitude = search.lat
            longitude = search.lng
            if latitude is None or longitude is None:
                raise ValueError("Coordinates not found.")
            return latitude, longitude
        except Exception as e:
            print(f"Error fetching coordinates for {placename}: {e}")
            return None, None

    # Tool: Filter DataFrame
    @tool
    def filter_df(conditions: dict) -> list[dict]:
        """
        Filter a DataFrame based on multiple conditions from user query provided as a dictionary.
        example parameter: conditions = {'year': 2020, 'poi_type': 'School'}
        """
        df = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")
        filtered_df = df.copy()
        for column, value in conditions.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
            else:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        return filtered_df
    
    def get_closest_points(lat, lon, num_points=5):
        """Find the `num_points` closest points to the given latitude and longitude."""
        df = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")

        def haversine(row):
            lat2, lon2 = row['LATITUDE'], row['LONGITUDE']
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat, lon, lat2, lon2])
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            r = 3963  # Radius of Earth in miles
            return r * c

        df['distance'] = df.apply(haversine, axis=1)
        closest_points = df.nsmallest(num_points, 'distance')
        return closest_points


    # CSV Agent Setup
    csv_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key),
        path="/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        extra_tools=[calculate_distance, get_geographic_coordinates],
        prefix=(
            "First filter out necessary rows using appropriate column values (e.g., year, poi_type, etc.). "
            "Based on the user query, come up with conditions dictionary. For example, conditions = {'year': 2020, 'poi_type': 'School'}"
            "If necessary, use the `calculate_distance` function to calculate distances between locations. "
            "Then, use the entire 290887 distance values for calculation afterward. NEVER use sample data. Read csv data from '/Users/sevinchnoori/Michelin4/processed_data/temp.csv'"
        )
    )

    # Store chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    bot_icon = "./michelin_mobility_intelligence_logo.jpeg"
    for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.image(bot_icon, width=30)  # Display chatbot icon
                    st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if "map" in prompt.lower():
            target_lat, target_lon = 34.070877749999994, -118.44685031405575
            closest_points = get_closest_points(target_lat, target_lon)

            # Create map and add markers
            map = Map({'LATITUDE': target_lat, 'LONGITUDE': target_lon})
            for _, point in closest_points.iterrows():
                map.marker(point, color='blue', icon='road', popup=True)

            # Display map
            map_html = map.get_map_html()
            components.html(map_html, height=500)
        else:
            response = csv_agent.invoke(prompt)
            placeholder = st.empty()
            full_response = response.get("output", "")
            placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
    # Display or clear chat messages
    # for message in st.session_state.messages:
    #     if message["role"] == "user":
    #         with st.chat_message("user"):
    #             st.write(message["content"])
    #     else:
    #         with st.chat_message("assistant"):
    #             st.image(bot_icon, width=30)  # Display chatbot icon
    #             st.write(message["content"])

    # def clear_chat_history():
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    # st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # def get_closest_points(lat, lon, num_points=5):
    #     """Find the `num_points` closest points to the given latitude and longitude."""
    #     df = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")

    #     def haversine(row):
    #         lat2, lon2 = row['LATITUDE'], row['LONGITUDE']
    #         lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat, lon, lat2, lon2])
    #         dlat = lat2_rad - lat1_rad
    #         dlon = lon2_rad - lon1_rad
    #         a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    #         c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    #         r = 3963  # Radius of Earth in miles
    #         return r * c

    #     df['distance'] = df.apply(haversine, axis=1)
    #     closest_points = df.nsmallest(num_points, 'distance')
    #     return closest_points
    
    # # User-provided prompt
    # replicate_api = True
    # if prompt := st.chat_input(disabled=not replicate_api):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.write(prompt)

    
    # # Generate a new response if the last message is not from the assistant
    # if st.session_state.messages[-1]["role"] != "assistant":
    #     with st.spinner("Thinking..."):
    #         response = csv_agent.invoke(prompt)
    #         placeholder = st.empty()
    #         full_response = response.get("output", "")

    #         if "map" in full_response.lower():
    #             target_lat, target_lon = 34.070877749999994, -118.44685031405575
    #             closest_points = get_closest_points(target_lat, target_lon)
    #             map = Map({'LATITUDE': target_lat, 'LONGITUDE': target_lon})

    #             for _, point in closest_points.iterrows():
    #                 map.marker(point, color='blue', icon='road', popup=True)

    #             map_html = map.get_map_html()
    #             components.html(map_html, height=500)
                
    #         else:
    #             placeholder.markdown(full_response)
    #             message = {"role": "assistant", "content": full_response}
    #             st.session_state.messages.append(message)
    #         # map a coordinate closest to this location (34.070877749999994,-118.44685031405575)