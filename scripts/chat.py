import os
import sys
import time
import openai
import threading
import numpy as np
import sounddevice as sd

import pyttsx3


from pydub import AudioSegment
from dotenv import load_dotenv


def thinking_animation(stop_event):
    while not stop_event.is_set():
        for i in range(3):
            sys.stdout.write("\rThinking" + "." * (i + 1))
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write("\r" + " " * 10 + "\r")  # Clear the line
        sys.stdout.flush()
        

SETTINGS = {
    "engine": "text-davinci-003",
    "max_tokens": 1024,
    "n": 1,
    "stop": None,
    "temperature": 0.7,
}

DURATION = 5 # duration of the sound recording in seconds
SAMPLE_RATE = 16000 # sample rate of the sound recording in Hz
FILE_PATH = "../recordings/" # path to the directory where the audio file will be saved
FILE_EXTENSION = ".mp3" # extension of the audio file


def record_audio(duration=5, sample_rate=16000):
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        blocking=True
    )
    return audio

def save_audio_to_mp3(audio, sample_rate, output_filename):
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    audio_segment.export(output_filename, format="mp3")
    
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def record_and_save_audio(duration=5, sample_rate=16000, output_filename="recording.mp3"):
    print(f"Recording audio for {duration} seconds...")
    audio = record_audio(duration, sample_rate)
    print(f"Saving audio to {output_filename}")
    save_audio_to_mp3(audio, sample_rate, output_filename)

def get_response(prompt):
    return openai.Completion.create(
        prompt=prompt,
        **SETTINGS
    )

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    # Initialize the OpenAI API client
    openai.api_key = api_key

    while True:
        # Get user input
        i = input("\nEnter your prompt for text chat or type 'r <filename>' to record an audio message (or type 'quit' to exit): ")

        if i.lower() == "quit":
            break
        
        if i.lower().split(" ")[0] == "r":
            filepath = FILE_PATH + f"{i.lower().split(' ')[1]}" + FILE_EXTENSION
            record_and_save_audio(duration=DURATION, sample_rate=SAMPLE_RATE, output_filename=filepath)
            audio_file= open(filepath, "rb")
            i = openai.Audio.transcribe("whisper-1", audio_file)
            i = i["text"].strip()

        # Start the thinking animation in a separate thread
        stop_event = threading.Event()
        animation_thread = threading.Thread(target=thinking_animation, args=(stop_event,))
        animation_thread.start()

        # Send a request to the API
        response = get_response(i)

        # Stop the thinking animation
        stop_event.set()
        animation_thread.join()

        # Extract the response and print it
        answer = response.choices[0].text
        print("Anser: " + answer)
        text_to_speech(answer)

if __name__ == "__main__":
    main()
