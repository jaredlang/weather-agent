# Project Goal

* Create a weather agent to get the current weather information and make an announcement like those on TV

## Known Issues

1. Text2Speech model doesn't know how to say "minus" if the temperature is below 0. (SOLVED)
    * Use prompt to have LLM add NAGATIVE in front of the temperature degree. 

2. Text2Speech doesn't have the local personality. (SOLVED)
    * SHOUT OUT [Coqui](https://github.com/coqui-ai/TTS). It is a great tts model. 
    * Next Step: Fine tune the TTS model to speak like a local reporter 

3. The temperature is always in Celsius. It should be in a local measure unit. (SOLVED)
    * Option 1 (traditional):  
        - Added the unit of measure to the prompt
        - Convert the value to fahrenheit for United State, Liberia, Myanmar
    * Option 2 (AI-way):  
        - Added the unit of measure to the tool parameters
        - LLM knows what parameter to pass in based on the UOM in that city 

4. Need to get other weather data besides the temperature. (DONE)

5. There is no image to show the weather data
    * Use the stabilityai/stable-diffusion-2-1 model to generate a city-view image. 
    * Next steps: use a smaller or specific model to generate a weather image or icon 

6. Add the error handling in the tools
    * Introduce the fallback chain or agent
    * Introduce retry 
    * Introduce try / catch / raise

7. Improve the performance 
    * On local CPU TTS takes 2-4 minutes and stable-diffusion takes 20-40 minutes. 
    * On P4000 or RTX6000, it takes half of that time. The cost is about $0.5/hr on Paperspace bare metal VM. 
    * To be tested on a hosted environment

8. Set up an async running model for web ui 
    * Start a process and notify the user when it is generated 
    * Store the historical inputs and outputs for the future reference

9. Add the review steps to the creation process
    * Allow human to review the output of each step 
    * Use AI agents

10. Implement multi-processing for text2speech and text2image. (DONE)
    * Use multiprocessing 
    