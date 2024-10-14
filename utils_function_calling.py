import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import pandas as pd
from geopy.distance import geodesic
import re

crash_data = pd.read_csv('processed_data/crash_data.csv')

def extract_additional_kwargs(response):
    content_str = response.content
    content_str = content_str.replace("\\", "")
    content_str = content_str.replace('"', "'")
    match = re.search(r"('function_call':\s*{[^}]*})", content_str)
    if match==None:
        return None, None
    additional_kwargs_str = match.group(0)

    start_index = additional_kwargs_str.find("'name':")
    end_index = additional_kwargs_str.find("'", start_index+9)
    function_name = additional_kwargs_str[start_index+9:end_index]

    # print(additional_kwargs_str)
    # print("function_name:", function_name)
    return additional_kwargs_str, function_name

def get_filtered_data(conditions):
    """Get fileterd data from crash_data based on the column names."""
    key, value = conditions.split(":")
    value = float(b=value)

    filtered_data = {
        "filtered_data": crash_data[crash_data[key]==value],
    }

    return json.dumps(filtered_data)

def get_stats(filtered_data, operation):
    """Get desired stats (e.g., mean, mode, count) from the filetered data."""

    if operation == "mean":
        result = filtered_data.mean()
    elif operation == "mode":
        result = filtered_data.mode()
    elif operation == "count":
        result = filtered_data.count()

    stats_info = {
        "stats_info": result,
    }

    return json.dumps(stats_info)

def get_user_gps(gps_avail):
    """Get the user's current location (latitude and longitude) based on the user's GPS."""

    if bool(gps_avail):
        # user_lat, user_long = utils.get_gps()
        user_lat = 50.0226593 # placeholders
        user_long = -0.43865 # placeholders
    else:
        user_lat = 34.0522 # placeholders
        user_long = -118.2437 # placeholders

    user_gps = {
        "latitude": user_lat,
        "longtitude": user_long,
    }

    return json.dumps(user_gps)

def get_closest_crash(latitude, longitude):
    """Find the closest crash to the user's GPS location."""
    distances = crash_data.apply(lambda row: geodesic((latitude, longitude), (row['latitude'], row['longitude'])).miles, axis=1)
    closest_crash = {
        "closest_crash": crash_data.iloc[distances.idxmin()],
    }
    return json.dumps(closest_crash)


def get_function_descriptions_multiple():
    function_descriptions_multiple = [
        {
            "name": "get_filtered_data",
            "description": "Get fileterd data from crash_data based on the conditions",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "string",
                        "description": "The dictionary of conditions to filter DataFrame. The format is {column_name:value,} e.g., {YEAR:2018}",
                    },
                },
                "required": ["conditions"],
            },
        },
        {
            "name": "get_stats",
            "description": "Get desired stats (e.g., mean, mode, count) from the filetered data",
            "parameters": {
                "type": "object",
                "properties": {
                    "filtered_data": {
                        "type": "string",
                        "description": "The filtered DataFrame",
                    },
                    "operation": {
                        "type": "string",
                        "description": "The type of operation to get the desired result (e.g., mean, mode, count)",
                    },
                },
                "required": ["filtered_data", "operation"],
            },
        },
        {
            "name": "get_user_gps",
            "description": "Get the user's current location (latitude and longitude) based on the user's GPS",
            "parameters": {
                "type": "object",
                "properties": {
                    "gps_avail": {
                        "type": "string",
                        "description": "The user's GPS availability (e.g. True, False)",
                    },
                },
                "required": ["gps_avail"],
            },
        },
        {
            "name": "get_closest_crash",
            "description": "Find the closest crash to the user's GPS location and output its information (LATITUDE,LONGITUDE,ARC_ID,YEAR,VH,CYC,PED,FATAL,SERIOUS_INJURY,MINOR_INJURY,NO_INJURY,nearest_poi_type,distance_to_poi)",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "string",
                        "description": "The user's GPS latitude",
                    },
                    "longitude": {
                        "type": "string",
                        "description": "The user's GPS longitude",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    ]
    return function_descriptions_multiple