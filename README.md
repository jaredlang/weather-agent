# Project Goal

* Create a weather agent to get the current weather information and make an announcement like those on TV

## Knownn Issues

1. Text2Speech model doesn't know how to say "minus" if the temperature is below 0. (SOLVED)
    * Use prompt to have LLM add NAGATIVE in front of the temperature degree. 
2. Text2Speech doesn't have the local personality. (SOLVED)
    * SHOUT OUT [Coqui](https://github.com/coqui-ai/TTS). It is a great tts model. 
    * Need to fine tune the speech model. 
3. The temperature is always in Celsius. It should be in a local measure unit
4. Need to get other weather data besides the temperature. (DONE)
5. There is no image to show the weather data
    * Use the stabilityai/stable-diffusion-2-1 model to generate an image. 
6. Add the error handling in the tools
