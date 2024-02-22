from dotenv import load_dotenv
from os import environ
import requests
from datetime import datetime

from diffusers import DiffusionPipeline
import io
from PIL import Image

import replicate

load_dotenv()

HUGGINGFACE_API_KEY = environ["HUGGINGFACE_API_KEY"]

USE_GPU = environ["USE_GPU"]

HF_TXT_TO_IMAGE_MODEL = "stabilityai/sdxl-turbo"
# "stabilityai/stable-diffusion-2-1"
HF_TXT_TO_IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
# "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# hosted on Replicate.com 
R8_TXT_TO_IMAGE_API_URL = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"

def generate_image_local(image_description: str): 
    pipeline = DiffusionPipeline.from_pretrained(HF_TXT_TO_IMAGE_MODEL)
    result = pipeline(image_description, num_inference_steps=50)
 
    image = result.images[0]

    pipeline = None

    return image


def generate_image_hf(image_description: str): 
    headers = {"Authorization": F"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(HF_TXT_TO_IMAGE_API_URL, headers=headers, json={
        "inputs": image_description + ". Make the image more cartoonish.",
    })

    if response.ok: 
        image_bytes = response.content
        # You can access the image with PIL.Image for example
        image = Image.open(io.BytesIO(image_bytes))
        return image    
    else:
        raise Exception(response.content)


# function to use stability-ai model to generate image
def generate_image_repl(prompt: str) -> str:
    output = replicate.run(
        R8_TXT_TO_IMAGE_API_URL,
        input={
            "prompt": prompt
        }
    )
    print("REPLLICATE OUTPUT: ", output)

    if output and len(output) > 0:
        # Get the image URL from the output
        image_url = output[0]
        print(f"generated image for {prompt}: {image_url}")

        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image_bytes = response.content
            # You can access the image with PIL.Image for example
            image = Image.open(io.BytesIO(image_bytes))
            return image
        else:
            raise Exception("Failed to download and save the image.")
    else:
        raise Exception("Failed to generate the image.")


def text2image(image_description: str, output_folder: str) -> str: 

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    image_format = "png"
    image_file_path = f"{output_folder}/image-{ts}.{image_format}"

    image = None
    if USE_GPU == "1": 
        print("generating image locally...")
        image = generate_image_local(image_description)
    else:
        print("generating image on Saas...")
        image = generate_image_repl(image_description)
    
    if image == None: 
        raise Exception("Error: no image generated")
    
    image.save(image_file_path, format=image_format)
    
    return image_file_path
