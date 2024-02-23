import replicate
import requests 
from datetime import datetime 
from os import environ

OUTPUT_FOLDER = environ["OUTPUT_FOLDER"]


def format_filename_or_dir(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.
 
Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.
 
"""
    import string 
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename


# function to use stability-ai model to generate image
def create_image(prompt: str) -> str:
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={
            "prompt": prompt
        }
    )

    if output and len(output) > 0:
        # Get the image URL from the output
        image_url = output[0]
        print(f"generated image for {prompt}: {image_url}")

        # Download the image and save it with a filename based on the prompt and current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        shortened_prompt = format_filename_or_dir(prompt[:50])
        image_file_path = f"{OUTPUT_FOLDER}/{shortened_prompt}_{current_time}.png"

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_file_path, "wb") as file:
                file.write(response.content)
            print(f"Image saved as '{image_file_path}'")
            return image_file_path
        else:
            raise Exception("Failed to download and save the image.")
    else:
        raise Exception("Failed to generate the image.")


def review_image(image_file_path: str, prompt: str):
    output = replicate.run(
        "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
        input={
            "image": open(image_file_path, "rb"),
            "prompt": f"What is happening in the image? From scale 1 to 10, decide how similar the image is to the text prompt {prompt}?",
        }
    )

    result = ""
    for item in output:
        result += item

    print("CRITIC : ", result)

    return result
