import pyaudio
from vosk import Model, KaldiRecognizer
import json
import os

# Vosk setup
vosk_model = Model("vosk_model")
recognizer = KaldiRecognizer(vosk_model, 48000)

# Audio setup
MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))  # Update to PyAudio index
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, input_device_index=MIC_SOURCE_ID, frames_per_buffer=16384)
stream.start_stream()

print("Listening... (say something, Ctrl+C to stop)")
while True:
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print("You said:", result.get("text", ""))
