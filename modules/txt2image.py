from os import environ
import requests
import time

from diffusers import DiffusionPipeline
import io
from PIL import Image

HUGGINGFACE_API_KEY = environ["HUGGINGFACE_API_KEY"]

OUTPUT_FOLDER = environ["OUTPUT_FOLDER"]

HF_TXT_TO_IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"
HF_TXT_TO_IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
# "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"


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


def text2image(text, return_dict = None): 

    ts = time.time()
    image_format = "png"
    image_file_path = f"{OUTPUT_FOLDER}/image-{ts}.{image_format}"

    image = generate_image_df(text)
    
    image.save(image_file_path, format=image_format)
    
    if return_dict is not None: 
        return_dict["image"] = image_file_path

    return image_file_path
