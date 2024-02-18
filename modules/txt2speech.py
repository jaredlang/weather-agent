from os import environ
import requests
import time

from TTS.api import TTS

HUGGINGFACE_API_KEY = environ["HUGGINGFACE_API_KEY"]

VOICE_SAMPLE = environ["VOICE_SAMPLE"]

OUTPUT_FOLDER = environ["OUTPUT_FOLDER"]

HF_TXT_TO_SPEECH_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
HF_TXT_TO_SPEECH_API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
# "https://api-inference.huggingface.co/models/microsoft/speecht5_tts" # Internal Server Error
# "https://api-inference.huggingface.co/models/suno/bark"  # Requires a paid license
# "https://api-inference.huggingface.co/models/facebook/mms-tts-eng" # Voice is not clear


# txt2speech (The best speech)
def txt2speech_tts(text, return_dict = None): 

    tts = TTS(HF_TXT_TO_SPEECH_MODEL, gpu=False)

    ts = time.time()
    file_path = f"{OUTPUT_FOLDER}/story_Coqui-{ts}.wav"
    
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=text,
                    file_path=file_path,
                    speaker_wav=VOICE_SAMPLE,
                    language="en")

    if return_dict is not None: 
        return_dict["audio"] = file_path

    return file_path


def txt2speech_Saas(text, return_dict = None): 
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payload = { "inputs": text }

    response = requests.post(HF_TXT_TO_SPEECH_API_URL, headers=headers, json=payload)
    if ("error" in response):
        print(response.error)
    else:
        ts = time.time()
        file_path = f"{OUTPUT_FOLDER}/announcement-{ts}.wav"
        with open(file_path, "wb") as f: 
            f.write(response.content)

    if return_dict is not None: 
        return_dict["audio"] = file_path

    return file_path


def txt2speech(text, return_dict = None):

    return txt2speech_tts(text, return_dict)
