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

import os
from dotenv import load_dotenv
import requests
import time

import streamlit as st 

load_dotenv()
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

OUTPUT_FOLDER = "./output"

# Define a function as a tool with decorator 
@tool
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


@tool 
def get_current_temperature(city: str) -> int:
    """Retrieves the current temperature for a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city to get the temperature for.

    Returns:
        float: The current temperature in degrees Celsius. Returns None if an error occurs.
    """

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}  # Use metric for Celsius

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for error status codes

        data = response.json()
        print("WEATHER DATA: ", json.dumps(data))

        temperature_kelvin = data["main"]["temp"]
        return math.floor(temperature_kelvin)

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


@tool 
def get_weather_report(city: str) -> str: 
    """Retrieves the current weather report for a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city to get the temperature for.

    Returns:
        float: The current weather report. Returns None if an error occurs.
    """

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}  # Use metric for Celsius

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for error status codes

        data = response.json()
        print("WEATHER DATA: ", json.dumps(data))

        weather_main_array = ". ".join([x['main'] for x in data['weather']])
        weather_description_array = ". ".join([x['description'] for x in data['weather']])

        report = (
            f"Weather at this hour is {weather_main_array} {weather_description_array}", 
            f"Current temperature is {math.ceil(data['main']['temp'])}. It could feel like {math.ceil(data['main']['feels_like'])} because of humidity",
            f"Humidity is {data['main']['humidity']} percent. ",
            "It is very humid." if data['main']['humidity'] > 75 else "It is a dry day." if data['main']['humidity'] < 25 else "It feels pretty comfortable. ",
            f"Today's temperature can go as high as {math.ceil(data['main']['temp_max'])} degrees. At night the temperature can drop to {math.floor(data['main']['temp_min'])} degrees. ",
            f"The rain amount is {data['rain']['1h']} in the unit of inch. " if 'rain' in data.keys() else "", 
            "Strong Wind." if data['wind']['speed'] > 10 else "The wind is calm." 
            # json_data["wind"]["deg"]
        )

        return report

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


#tools = [get_current_temperature, celsius_to_fahrenheit]

tools = [get_weather_report]

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very powerful assistant, but don't know current events"), 
    ("user", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])

# Define a factual LLM 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

    input = f"what's the current temperature in {place}? Always use the local unit of measurement."
    answer = agent_executor.invoke({"input": input})

    print("ANSWER: ", answer)

    audio_file_path = txt2speech(answer["output"])
    print("AUDIO: ", audio_file_path)

    return {
        "text": answer["output"], 
        "audio": audio_file_path
    }


def test():
    #answer = agent_executor.invoke({"input": "what's the current temperature in London?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})

    # answer = agent_executor.invoke({"input": "what's the weather like in Houston?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in Houston? Use Fahrenheit. "})

    # answer = agent_executor.invoke({"input": "what's the weather like in Atlanta?"})
    # answer = agent_executor.invoke({"input": "what's the weather like in Chicago?"})
    answer = agent_executor.invoke({"input": "what's the weather like in Orlando?"})

    print("ANSWER: ", answer)

    prompt = ChatPromptTemplate.from_template(""" You are an experienced weather reporter in WHBC network. 
                                              Give a vivid and detailed description in a casual tone based on the following weather report: 
                                              WEATHER REPORT:  {weather_report} """)

    chain = {"weather_report": RunnablePassthrough()} | prompt | ChatOpenAI() | StrOutputParser()

    description = chain.invoke(answer)

    print("DESCRIPTIION: ", description)

    description_excerpt = "\n\n".join(description.split("\n\n")[:3])

    audio_file_path = txt2speech(description_excerpt)
    print("AUDIO: ", audio_file_path)


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
    test()
    # app()
