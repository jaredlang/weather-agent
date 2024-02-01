import math
from langchain_openai import ChatOpenAI
from langchain.agents import tool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser
)
from langchain.agents import AgentExecutor

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
        temperature_kelvin = data["main"]["temp"]
        return math.floor(temperature_kelvin)

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

tools = [get_current_temperature, celsius_to_fahrenheit]

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

    audio_file_path = txt2speech_Saas(answer["output"])
    print("AUDIO: ", audio_file_path)

    return {
        "text": answer["output"], 
        "audio": audio_file_path
    }


def test():
    #answer = agent_executor.invoke({"input": "what's the current temperature in London?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})

    answer = agent_executor.invoke({"input": "what's the current temperature in Houston?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in Houston? Use Fahrenheit. "})

    print("ANSWER: ", answer)

    audio_file_path = txt2speech_Saas(answer["output"])
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
    # test()
    app()
