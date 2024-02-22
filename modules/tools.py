from dotenv import load_dotenv
import math
import json
import requests 
from os import environ

from langchain.agents import tool 

load_dotenv()

OPENWEATHER_API_KEY = environ["OPENWEATHER_API_KEY"]

OPENWEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


# Define an internal function for the tools to call 
def celsius_to_fahrenheit(celsius) -> float:
  """
  Converts a temperature in Celsius to Fahrenheit.

  Args:
      celsius: The temperature in degrees Celsius.

  Returns:
      The temperature in degrees Fahrenheit.
  """
  fahrenheit = math.floor((celsius * 9/5) + 32)
  return fahrenheit


# Define an internal function for the tools to call
def get_raw_weather_data(city: str, units: str = "metric") -> str:
    """Retrieves the current weather data in a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city to get the weather for.
        units (str): use metric for Celsius, imperial for Fahrenheit.

    Returns:
        int: The current weather in degrees Celsius. Returns None if an error occurs.
    """
    
    params = {
      "appid": OPENWEATHER_API_KEY, 
      "q": city,
      "units": units
    }  

    try:
        response = requests.get(OPENWEATHER_API_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for error status codes

        data = response.json()
        print("WEATHER DATA: ", json.dumps(data))

        structured_data = {
            "city": city, 
            "country_code": data['sys']['country'], 
            "summary":  ". ".join([x['main'] for x in data['weather']]), 
            "description": ". ".join([x['description'] for x in data['weather']]), 
            "temp": data["main"]["temp"], 
            "feels_like": data['main']['feels_like'], 
            "temp_max": data['main']['temp_max'], 
            "temp_min": data['main']['temp_min'], 
            "humidity": data['main']['humidity'], 
            "visibility": data["visibility"], 
            "wind_speed": data['wind']['speed'], 
            # data["wind"]["deg"]
        }

        if 'rain' in data.keys(): 
            structured_data['rain_1h'] = data['rain']['1h']

        return structured_data

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        raise Exception(f"Weather data not available in the city {city}")
    

@tool 
def get_current_temperature(city: str, units: str) -> int:
    """Retrieves the current temperature in a given city.

    Args:
        city (str): The name of the city to get the temperature for.
        units (str): use metric for Celsius, imperial for Fahrenheit.

    Returns:
        int: The current temperature in degrees. Returns None if an error occurs.
    """

    data = get_raw_weather_data(city, units)

    temperature_kelvin = data['temp']
    return math.floor(temperature_kelvin)
    

@tool 
def get_weather_summary(city: str, units: str) -> str: 
    """Retrieves a brief summary of the current weather in a given city.

    Args:
        city (str): The name of the city to get a brief weather report for.
        units (str): use metric for Celsius, imperial for Fahrenheit.

    Returns:
        str: The weather summary. Returns None if an error occurs.
    """

    data = get_raw_weather_data(city, units)

    # format the response
    summary = f"Weather is {data['summary']}, {data['description']}."

    return summary 

    
@tool 
def get_weather_detail(city: str, units: str) -> str: 
    """Retrieves the detailed weather report for a given city.

    Args:
        city (str): The name of the city to get a detailed weather report for.
        units (str): use metric for Celsius, imperial for Fahrenheit.

    Returns:
        str: The weather detail. Returns None if an error occurs.
    """

    data = get_raw_weather_data(city, units)

    # After adding the units parameter to the get_raw_weather_data tool, 
    # LLM knows what unit of measure depending on the city location. 
    # SMART! 
    # 
    # Just specify the unit of measure 
    uom = "fahrenheit" if units == "imperial" else "celsius" 
    # - No need for a manual conversion. 
    # For United State, Liberia, Myanmar, convert the temp to fahrenheit
    # uom = "celsius" # by default
    # if units == "imperial": 
    #     uom = "fahrenheit"
    #     data['temp'] = celsius_to_fahrenheit(data['temp'])
    #     data['feels_like'] = celsius_to_fahrenheit(data['feels_like'])
    #     data['temp_max'] = celsius_to_fahrenheit(data['temp_max'])
    #     data['temp_min'] = celsius_to_fahrenheit(data['temp_min'])

    # round up 
    data['temp'] = math.ceil(data['temp'])
    data['feels_like'] = math.ceil(data['feels_like'])
    data['temp_max'] = math.ceil(data['temp_max'])
    data['temp_min'] = math.ceil(data['temp_min'])

    # format the response
    report = (
        f"Weather in {city}, {data['country_code']} at this hour is {data['summary']}, {data['description']}.", 
        f"Current temperature is {data['temp']} degrees {uom}.", 
        f"It could feel like {data['feels_like']} degrees {uom}.",
        f"Humidity is {data['humidity']} percent. ",
        "It is very humid." if data['humidity'] > 75 else "It is a dry day." if data['humidity'] < 25 else "It feels pretty comfortable.",
        f"In the daytime the temperature rises as high as {data['temp_max']} degrees {uom}.",
        f"At night the temperature drops to {data['temp_min']} degrees {uom}.",
        f"The rain amount is {data['rain_1h']} in the unit of inch. " if 'rain_1h' in data.keys() else "", 
        "Strong Wind." if data['wind_speed'] > 10 else "The wind is calm.",
        "Visibility is poor" if data["visibility"] < 2500 else ""
    )

    return report 

