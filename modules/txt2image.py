from os import environ
import requests
from datetime import datetime

from diffusers import DiffusionPipeline
import io
from PIL import Image

import replicate

HUGGINGFACE_API_KEY = environ["HUGGINGFACE_API_KEY"]

OUTPUT_FOLDER = environ["OUTPUT_FOLDER"]

HF_TXT_TO_IMAGE_MODEL = "stabilityai/sdxl-turbo"
# "stabilityai/stable-diffusion-2-1"
HF_TXT_TO_IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
# "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# hosted on Replicate.com 
R8_TXT_TO_IMAGE_API_URL = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"

def generate_image_df(image_description: str): 
    pipeline = DiffusionPipeline.from_pretrained(HF_TXT_TO_IMAGE_MODEL)
    result = pipeline(image_description, num_inference_steps=50)

    image = result.images[0]

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


def text2image(image_description: str) -> str: 

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    image_format = "png"
    image_file_path = f"{OUTPUT_FOLDER}/image-{ts}.{image_format}"

    image = generate_image_repl(image_description)
    
    image.save(image_file_path, format=image_format)
    
    return image_file_path
