import json
import pyaudio
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from piper.voice import PiperVoice
from openai import OpenAI
import os

# Vosk STT setup
vosk_model = Model("vosk_model")
recognizer = KaldiRecognizer(vosk_model, 48000)

# PipeWire audio setup
MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))  # USB mic
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, input_device_index=MIC_SOURCE_ID, frames_per_buffer=16384)
stream.start_stream()

# Piper TTS setup
piper_model_path = "piper_voice/en_US-lessac-medium.onnx"
voice = PiperVoice.load(piper_model_path)

# LLM API setup
client = OpenAI(api_key="sk-c36ea8e4c694498c9bdfd98ac3350d7e", base_url="https://api.deepseek.com/v1")

def listen():
    print("Listening... (say 'exit' to quit)")
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                return text

def speak(text):
    print("Speaking:", text)
    audio_stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    audio_stream.start()
    for audio_chunk in voice.synthesize(text):
        int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        audio_stream.write(int_data)

    audio_stream.stop()
    audio_stream.close()

def get_llm_response(user_input):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content.strip()

# Main loop
speak("Hello, I'm your AI assistant. Start talking!")
while True:
    user_text = listen()
    print("You said:", user_text)
    if "exit" in user_text.lower() or "quit" in user_text.lower():
        speak("Goodbye!")
        break
    llm_reply = get_llm_response(user_text)
    speak(llm_reply)
