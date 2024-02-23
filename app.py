from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser
)
from langchain.agents import AgentExecutor

import os 
from dotenv import load_dotenv
import time

from modules.tools import get_weather_summary, get_weather_detail
from modules.txt2speech import txt2speech
from modules.txt2image import text2image

import multiprocessing

import streamlit as st 

load_dotenv()

#OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"]

# Define a factual LLM 
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# create weather agent
def create_weather_agent(): 

    # Define a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a very powerful assistant, but don't know current events"), 
        ("user", "{input}"), 
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])

    # #tools = [get_current_temperature, celsius_to_fahrenheit]
    tools = [get_weather_detail, get_weather_summary]

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

    return agent_executor


agent_executor = create_weather_agent()


def txt2speech_subproc(text: str, output_folder: str, return_dict=None) -> str: 
    audio_file_path = txt2speech(text, output_folder)
    if return_dict is not None: 
        return_dict["audio"] = audio_file_path


def txt2image_subproc(text: str, output_folder: str, return_dict=None) -> str: 
    image_file_path = text2image(text, output_folder)
    if return_dict is not None: 
        return_dict["image"] = image_file_path
    

def create_report(place):

    output_folder = os.path.join(OUTPUT_FOLDER, place)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #answer = agent_executor.invoke({"input": "what's the current temperature in London?"})
    #answer = agent_executor.invoke({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})
    #answer = agent_executor.invoke({"input": "what's the current temperature in Houston? Use Fahrenheit. "})
    report = agent_executor.invoke({"input": f"Get a detailed weather report for the city of {place}?"})
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

    # prompt1 = ChatPromptTemplate.from_template(
    #     f"Get a brief summary of weather report for the city {place}."
    # )
    # chain1 = {"place": RunnablePassthrough()} | prompt1 | llm_with_tools | StrOutputParser()

    summary = agent_executor.invoke({"input": f"Get a weather summary for the city of {place}. It must be less than 15 words. Do NOT repeat any word in the input."})
    print("SUMMARY for image:", summary)

    # Use multiprocess to start both text2speech and text2image
    # Use a shared variable to communicate
    proc_manager = multiprocessing.Manager()
    return_dict = proc_manager.dict()

    # create an audio 
    ttsph_process = multiprocessing.Process(
        target=txt2speech_subproc, 
        args=(description, output_folder, return_dict)
    )
    # audio_file_path = txt2speech(description)
    # print("AUDIO: ", audio_file_path)

    # create an image 
    # text2image doesn't take or require a lengthy description 
    ttimg_process = multiprocessing.Process(
        target=txt2image_subproc, 
        args=(summary["output"], output_folder, return_dict)
    )
    # image_file_path = text2image(summary["output"])
    # print("IMAGE: ", image_file_path)    

    processes = [ttsph_process, ttimg_process]

    for proc in processes: 
        proc.start()

    for proc in processes:
        proc.join()

    print(return_dict)

    return {
        "summary": summary["output"], 
        "detail": description, 
        "audio": return_dict["audio"], 
        "image": return_dict["image"], 
    }


def autogen_image_gen(): 
    from modules.autogen_agent import user_proxy, manager

    # # Start the conversation
    chat_result = user_proxy.initiate_chat(
        manager, message="Houston city scenary at 6:30pm, Sunny sky with few clouds, strong wind, nobody walks")
    
    print("CHAT_RESULT: ", chat_result)

    # get the last generated image, which is the best image 
    last_image_file_path = None
    for history in chat_result.chat_history:
        history_keys = history.keys() 
        if "content" in history_keys and "name" in history_keys and "role" in history_keys: 
            if history["name"] == "create_image" and history["role"] == "function":
                last_image_file_path = history["content"]
        # if "name" in history_keys and "role" in history_keys and "function_call" in history_keys: 
        #     if history["name"] in ["graphic_designer", ""] and history["role"] == "assistant":
        #         func_call = history["function_call"]
        #         if "arguments" in func_call.keys():
        #             func_call_arguments = func_call["arguments"]
        #             if isinstance(func_call_arguments, str): 
        #                 func_call_arguments = json.loads(func_call["arguments"])
        #                 if "image_file_path" in func_call_arguments.keys(): 
        #                     last_image_file_path = func_call_arguments["image_file_path"]

    print("Best Image: ", last_image_file_path)

    return last_image_file_path


def app(): 
    st.set_page_config(page_title="AI Weather Agent", page_icon=":bird:")

    st.header("AI Weather Agent :bird:")
    place = st.text_input("Enter a place:", help="Any place in the world", placeholder="Any place in the world")

    placeholder = st.empty()

    if place:
        start_time = time.time()
        placeholder.text(f"Checking the weather ...")

        report = create_report(place)

        end_time = time.time()
        agent_response_time = int(end_time - start_time)

        placeholder.text(f"The current weather is:")

        with st.expander("Summary"):
            st.write(report["summary"])

        with st.expander("Detail"):
            st.write(report["detail"])

        with st.expander(f"Audio"):
           st.audio(report["audio"])

        with st.expander(f"Image"):
           st.image(report["image"])            

        st.text(f"Running time: {agent_response_time} seconds")


if __name__ == "__main__": 
    # Atlanta, Orlando, Houston, New York, Calgary, Stockholm, Seattle
    # ABC, XYZ - negative testing
    # create_report("Atlanta")
    autogen_image_gen()
    # app()
