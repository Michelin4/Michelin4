# %%
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from getpass import getpass
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import geocoder
import ast

api_key = getpass("Enter your OpenAI API key: ")

# %%
df_crash = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")
df_acceleration = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/harsh_accel_data.csv")
df_braking = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/harsh_braking_data.csv")
df_crash.head()

# %%
df = df_crash

# %%
from typing import List, Dict

# Tool: Calculate Distance
@tool
def calculate_distance(lat1: float, lon1: float) -> List[Dict]:
    """
    Calculate the distance in miles between a given point (lat1, lon1) and 
    each row's geographic point using the Haversine formula. Add a `distance` column.
    """
    # if isinstance(df, str):
    #     df = pd.DataFrame(ast.literal_eval(df))  # Safely parse string to DataFrame
    # elif isinstance(df, dict):
    #     df = pd.DataFrame(df)
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
def filter_df(conditions: dict) -> List[Dict]:
    """
    Filter a DataFrame based on multiple conditions from user query provided as a dictionary.
    example parameter: conditions = {'year': 2020, 'poi_type': 'School'}
    """
    # if isinstance(df, str):
    #     df = pd.DataFrame(ast.literal_eval(df))  # Safely parse string to DataFrame
    # elif isinstance(df, dict):
    #     df = pd.DataFrame(df)
    df = pd.read_csv("/Users/sevinchnoori/Michelin4/processed_data/crash_data.csv")

    filtered_df = df.copy()
    for column, value in conditions.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column] == value]
        else:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    return filtered_df

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

# %% [markdown]
# Info for the nearest crash to UCLA
# How many crashes happened in 2020 within 20 miles from Hollywood?

# %%
query = "How many crashes happened in 2020 within 20 miles from Hollywood?"
response = csv_agent.invoke(query)
print("Question:", response.get("input"))
print("Response:", response.get("output"))


