import json
import pyaudio
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from piper.voice import PiperVoice
import os
import ollama
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables will be set inside the main function
vosk_model = None
recognizer = None
voice = None
p = None
stream = None

def speak(text):
    """Converts text to speech and plays the audio."""
    if not text:
        return
    logging.info(f"Speaking: {text}")
    try:
        audio_stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
        audio_stream.start()
        for audio_chunk in voice.synthesize(text):
            int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            audio_stream.write(int_data)
        audio_stream.stop()
        audio_stream.close()
    except Exception as e:
        logging.error(f"Error playing audio: {e}")

def listen():
    """Listens for a user's voice and returns the recognized text."""
    logging.info("Listening... (say 'exit' to quit)")
    while True:
        try:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    return text
        except Exception as e:
            logging.error(f"Error during listening: {e}")
            return ""

def get_llm_response(user_input):
    """Sends user input to Ollama and returns the response."""
    try:
        logging.info(f"Sending to Ollama: {user_input}")
        response = ollama.chat(model='tinyllama', messages=[
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return "I'm sorry, I couldn't process that request."
        
def clean_response(text):
    """Removes common markdown and formatting characters from a string."""
    # This function was not in the original code but is in your test file.
    # I've added a basic implementation for completeness.
    if not isinstance(text, str):
        return ""
    text = text.replace('**', '').replace('*', '').replace('`', '').replace('__', '').replace('#', '')
    return text.strip()

def main():
    """Initializes the assistant and starts the main loop."""
    global vosk_model, recognizer, voice, p, stream

    # --- 1. Vosk STT setup ---
    try:
        vosk_model_path = "vosk_model" 
        vosk_model = Model(vosk_model_path)
        recognizer = KaldiRecognizer(vosk_model, 48000)
    except Exception as e:
        logging.error(f"Error loading Vosk model: {e}")
        logging.error("Please check the model path and ensure the folder is present.")
        return # Gracefully exit the main function instead of the program

    # --- 2. Piper TTS setup ---
    try:
        piper_model_path = "piper_voice/en_US-lessac-medium.onnx"
        voice = PiperVoice.load(piper_model_path)
    except Exception as e:
        logging.error(f"Error loading Piper model: {e}")
        logging.error("Please check the model path and ensure the folder is present.")
        return

    # --- 3. Audio & STT Loop Setup ---
    try:
        MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, input_device_index=MIC_SOURCE_ID, frames_per_buffer=16384)
        stream.start_stream()
    except Exception as e:
        logging.error(f"Error setting up audio stream: {e}")
        return

    # --- 4. Main Conversation Loop ---
    speak("Hello, I'm your AI assistant. Start talking!")
    while True:
        user_text = listen()
        logging.info(f"You said: {user_text}")
        if "exit" in user_text.lower() or "quit" in user_text.lower():
            speak("Goodbye!")
            break
        llm_reply = get_llm_response(user_text)
        llm_reply = clean_response(llm_reply)
        speak(llm_reply)

if __name__ == "__main__":
    main()