import math
import json 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool 
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser
)
from langchain.agents import AgentExecutor

from TTS.api import TTS

from diffusers import DiffusionPipeline
import io
from PIL import Image

import os
from dotenv import load_dotenv
import requests
import time

import streamlit as st 

load_dotenv()
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

OUTPUT_FOLDER = "./output"

OPENAI_MODEL = "gpt-3.5-turbo"

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
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
      "appid": OPENWEATHER_API_KEY, 
      "q": city,
      "units": units
    }  

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for error status codes

        data = response.json()
        print("WEATHER DATA: ", json.dumps(data))

        structured_data = {
            "city": city, 
            "country_code": data['sys']['country'], 
            "overview":  ". ".join([x['main'] for x in data['weather']]), 
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
    """Retrieves a brief summary of the current weather in a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city to get a brief weather report for.
        units (str): use metric for Celsius, imperial for Fahrenheit.

    Returns:
        str: The weather summary. Returns None if an error occurs.
    """

    data = get_raw_weather_data(city, units)

    # format the response
    summary = f"Weather is {data['overview']}, {data['description']}."

    return summary 

    
@tool 
def get_weather_detail(city: str, units: str) -> str: 
    """Retrieves the detailed weather report for a given city using the OpenWeatherMap API.

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
        f"Weather in {city}, {data['country_code']} at this hour is {data['overview']}, {data['description']}.", 
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


#tools = [get_current_temperature, celsius_to_fahrenheit]
tools = [get_weather_detail, get_weather_summary]

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very powerful assistant, but don't know current events"), 
    ("user", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])

# Define a factual LLM 
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# Bind LLM with tools
llm_with_tools = llm.bind_tools(tools)

# Create an agent 
agent = (
    {
        "input": lambda x: x["input"], 
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        )
    }
    | prompt 
    | llm_with_tools
    #| llm
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# txt2speech (The best speech)
def txt2speech(text): 

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    ts = time.time()
    file_path = f"{OUTPUT_FOLDER}/story_Coqui-{ts}.wav"
    
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=text,
                    file_path=file_path,
                    speaker_wav=r"./voice_samples/samples_en_sample.wav",
                    language="en")

    return file_path


def txt2speech_Saas(text): 
    #API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts" # Internal Server Error
    #API_URL = "https://api-inference.huggingface.co/models/suno/bark"  # Requires a paid license
    #API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng" # Voice is not clear
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payload = { "inputs": text }

    response = requests.post(API_URL, headers=headers, json=payload)
    if ("error" in response):
        print(response.error)
    else:
        ts = time.time()
        file_path = f"{OUTPUT_FOLDER}/announcement-{ts}.wav"
        with open(file_path, "wb") as f: 
            f.write(response.content)

    return file_path


def create_announcement(place): 

    input = f"what's the current temperature in {place}?"
    answer = agent_executor.invoke({"input": input})

    print("ANSWER: ", answer)

    audio_file_path = txt2speech(answer["output"])
    print("AUDIO: ", audio_file_path)

    return {
        "text": answer["output"], 
        "audio": audio_file_path
    }


def generate_image_df(image_description: str): 
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    result = pipeline(image_description, num_inference_steps=50)
    image = result.images[0]

    return image


def generate_image_hf(image_description: str): 
    # Huggingface model: 
    # API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": F"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(API_URL, headers=headers, json={
        "inputs": image_description + ". Make the image more cartoonish.",
    })

    if response.ok: 
        image_bytes = response.content
        # You can access the image with PIL.Image for example
        image = Image.open(io.BytesIO(image_bytes))
        return image
    
    else:
        raise Exception(response.content)


def text2image(text): 

    ts = time.time()
    image_format = "png"
    image_file_path = f"{OUTPUT_FOLDER}/image-{ts}.{image_format}"

    image = generate_image_df(text)
    
    image.save(image_file_path, format=image_format)
    
    return image_file_path


def test(place):

    #answer = agent_executor.invoke({"input": "what's the current temperature in London?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})
    #answer = agent_executor.invoke({"input": "what's the current temperature in Houston? Use Fahrenheit. "})
    report = agent_executor.invoke({"input": f"Get a detailed weather report for the city {place}?"})
    print("REPORT: ", report)

    #!!! Use prompt to make sure the announcement doesn't miss negative before temperature. 
    #    Otherwise, TTS just says the numeric value and ignore negative. 
    # Too many instructions in the prompt makes LLM unpredictable. Somtimes it ignores some of them. 
    #   If a temperature is negative, include NEGATIVE before the temperature numeric value. 
    #   Do NOT say Celsius or Fahrenheit after temperature.
    prompt2 = ChatPromptTemplate.from_template(
        """ You are an experienced weather reporter and give daily weather updates on WHBC TV Network. 
            Give a vivid and detailed description in a casual tone based on the following weather report. 
            If a temperature is negative, say NEGATIVE before the temperature numeric value. 
            WEATHER REPORT:  
            {weather_report} 
        """)

    chain2 = {"weather_report": RunnablePassthrough()} | prompt2 | llm | StrOutputParser()

    description = chain2.invoke(report)
    print("DESCRIPTIION for audio: ", description)
    #description_excerpt = "\n\n".join(description.split("\n\n")[:3])

    audio_file_path = txt2speech(description)
    print("AUDIO: ", audio_file_path)

    # create an image 
    # text2image doesn't take or require a lengthy description 
    prompt1 = ChatPromptTemplate.from_template(
        f"Get a brief summary of weather report for the city {place}."
    )
    #chain1 = {"place": RunnablePassthrough()} | prompt1 | llm_with_tools | StrOutputParser()

    summary = agent_executor.invoke({"input": f"Get a weather summary for the city {place}. It must be less than 15 words. Do NOT repeat any word in the input."})
    print("SUMMARY for image:", summary)

    image_file_path = text2image(summary["output"])
    print("IMAGE: ", image_file_path)    


def app(): 
    st.set_page_config(page_title="AI Weather Agent", page_icon=":bird:")

    st.header("AI Weather Agent :bird:")
    place = st.text_input("Enter a place:", help="Any place in the world", placeholder="Any place in the world")

    placeholder = st.empty()

    if place:
        start_time = time.time()
        placeholder.text(f"Checking the weather ...")

        announcement = create_announcement(place)

        end_time = time.time()
        agent_response_time = int(end_time - start_time)

        placeholder.text(f"The current weather is:")

        with st.expander("Text"):
            st.write(announcement["text"])

        with st.expander(f"Audio"):
           st.audio(announcement["audio"])            

        st.text(f"running time: {agent_response_time} seconds")


if __name__ == "__main__": 
    # Atlanta, Orlando, Houston, New York, Calgary, Stockholm
    test("ABC")
    # app()
