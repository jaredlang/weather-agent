# Project Goal

* Create a weather agent to get the current weather information and make an announcement like those on TV

## New Features and Issues

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

5. There is no image to show the weather data (In Progress)
    * Use the stabilityai/stable-diffusion-2-1 model to generate a city-view image. 
    * Next steps: use a smaller or specific model to generate a weather image or icon 

6. Add the review steps to review/critic the generated content (In Progress)
    * Allow human to review the output of each step 
    * Use AutoGen and assistants (graphic designer and critic)
    * Run stability-ai/sdxl for image creation and yorickvp/llava for image review
    * Use a different model to create an image based on the prompt and a sample image 

7. Improve the performance (In Progress)
    * On local CPU TTS takes 2-4 minutes and stable-diffusion takes 20-40 minutes. 
    * On P4000 or RTX6000, it takes half of that time. The cost is about $0.5/hr on Paperspace bare metal VM. 
    * Implemented multi-processing for text2speech and text2image
    * To be tested on a hosted environment (Huggingface, Azure OpenAI or AWS Sagemaker)
    * replicate.com offers a performant run on some largest models (image run returns withn 20s and costs less than $0.01/hr per run)
    * To host the coqui-xtts-v2 model on replicate.com

8. Store the generated data for the future reference (To Do)
    * Store text/audio/image into blob storage 
    * Create UI to retrieve the stored content 

9. Set up an async running model for web ui (To Do)
    * Start a process and notify the user when it is generated 
    * Store the historical inputs and outputs for the future reference

10. Add the error handling in the tools (To Do)
    * Introduce the fallback chain or agent
    * Introduce retry 
    * Introduce try / catch / raise
