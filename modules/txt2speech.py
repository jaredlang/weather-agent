from dotenv import load_dotenv
from os import environ
import requests
from datetime import datetime

from TTS.api import TTS

load_dotenv()

HUGGINGFACE_API_KEY = environ["HUGGINGFACE_API_KEY"]

COQUI_VOICE_SAMPLE = environ["COQUI_VOICE_SAMPLE"]

USE_GPU = environ["USE_GPU"]

HF_TXT_TO_SPEECH_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
HF_TXT_TO_SPEECH_API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
# "https://api-inference.huggingface.co/models/microsoft/speecht5_tts" # Internal Server Error
# "https://api-inference.huggingface.co/models/suno/bark"  # Requires a paid license
# "https://api-inference.huggingface.co/models/facebook/mms-tts-eng" # Voice is not clear


# txt2speech (The best speech)
def txt2speech_tts(narrative: str, file_path: str) -> str: 

    tts = TTS(HF_TXT_TO_SPEECH_MODEL, gpu=(USE_GPU == "True"))
    
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=narrative,
                    file_path=file_path,
                    speaker_wav=COQUI_VOICE_SAMPLE,
                    language="en")

    return file_path


def txt2speech_Saas(narrative: str, file_path: str) -> str: 
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payload = { "inputs": narrative }

    response = requests.post(HF_TXT_TO_SPEECH_API_URL, headers=headers, json=payload)
    if ("error" in response):
        print(response.error)
    else:
        with open(file_path, "wb") as f: 
            f.write(response.content)

    return file_path


def txt2speech(narrative: str, output_folder: str) -> str:

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"{output_folder}/speech-{ts}.wav"

    return txt2speech_tts(narrative, file_path)
