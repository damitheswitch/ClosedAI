import json
import pyaudio
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from piper.voice import PiperVoice
from openai import OpenAI
import os
from dotenv import load_dotenv
import cv2 
from deepface import DeepFace 
import time
from pathlib import Path
import datetime

# Load environment variables from .env file
load_dotenv()

# Define paths using pathlib for better cross-platform compatibility
BASE_DIR = Path(__file__).parent
VOSK_MODEL_PATH = BASE_DIR / "vosk-model-small-en-us-0.15"
PIPER_VOICE_PATH = BASE_DIR / "piper_voice" / "en_US-lessac-medium.onnx"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))

def initialize_vosk():
    """Initializes the Vosk speech recognition model."""
    if not VOSK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
    vosk_model = Model(str(VOSK_MODEL_PATH))
    return KaldiRecognizer(vosk_model, 48000)

def initialize_piper():
    """Initializes the Piper text-to-speech voice."""
    if not PIPER_VOICE_PATH.exists():
        raise FileNotFoundError(f"Piper voice model not found at {PIPER_VOICE_PATH}")
    return PiperVoice.load(str(PIPER_VOICE_PATH))

def initialize_microphone():
    """Initializes the PyAudio microphone stream."""
    p = pyaudio.PyAudio()
    for attempt in range(5):
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, 
                             input=True, input_device_index=MIC_SOURCE_ID, 
                             frames_per_buffer=16384)
            stream.start_stream()
            print("Microphone opened successfully!")
            return stream, p
        except OSError as e:
            print(f"Attempt {attempt + 1} failed to open microphone: {e}")
            time.sleep(1)
    
    print("Failed to open microphone after 5 attempts. Exiting.")
    exit(1)

def detect_mood_from_face():
    """Detects mood from a facial expression using DeepFace."""
    
    # Use a fallback if a camera is not available
    mood = "neutral"
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Camera {camera_index} not accessible.")
                continue
            
            # Capture a single frame
            time.sleep(1) # Give camera time to adjust
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"Failed to capture frame from camera {camera_index}.")
                continue
            
            # Analyze the frame for emotion
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False, 
                silent=True,
                detector_backend='opencv'
            )
            
            if result and len(result) > 0:
                mood = result[0]['dominant_emotion']
                print(f"Detected mood: {mood}")
                break # Exit loop on successful detection
            
        except Exception as e:
            print(f"DeepFace analysis failed: {e}")
            
    # Final fallback if camera detection fails
    if mood == "neutral":
        print("Using fallback mood detection.")
        hour = datetime.datetime.now().hour
        if 6 <= hour < 12:
            mood = "happy"
        elif 12 <= hour < 18:
            mood = "neutral"
        elif 18 <= hour < 22:
            mood = "relaxed"
        else:
            mood = "sleepy"

    return mood

def clean_response(text):
    """
    Removes markdown formatting characters from the LLM's response.
    """
    return text.replace('*', '').replace('#', '').replace('`', '')


def get_llm_response(user_input, mood="neutral"):
    """Generates LLM response based on user input and mood."""
    mood_instructions = {
        "happy": "Respond enthusiastically and positively to match the user's happy mood.",
        "sad": "Respond empathetically and supportively to cheer up the sad user.",
        "angry": "Respond very calmly and carefully to de-escalate the angry user.",
        "neutral": "Respond normally.",
        "fear": "Respond reassuringly and comforting.",
        "surprise": "Respond with excitement and curiosity.",
        "disgust": "Respond with understanding and offer alternatives.",
        "relaxed": "Respond in a calm and peaceful manner.",
        "sleepy": "Respond gently and quietly."
    }.get(mood, "Respond normally.")
    
    system_prompt = f"You are a voice assistant. {mood_instructions} Give short, concise answers. Never use markdown, asterisks, hashes, or any formatting. Speak naturally like a human."
    
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    # Use the new clean_response function
    return clean_response(response.choices[0].message.content)

def listen(stream, recognizer):
    """Listens for user input and returns transcribed text."""
    print("Listening... (say 'exit' to quit)")
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                return text

def speak(text, voice):
    """Synthesizes text to speech using Piper."""
    print("Speaking:", text)
    audio_stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    with audio_stream:
        for audio_chunk in voice.synthesize(text):
            int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            audio_stream.write(int_data)

def main():
    """Main function to run the voice assistant."""
    try:
        recognizer = initialize_vosk()
        voice = initialize_piper()
        stream, p = initialize_microphone()
    except FileNotFoundError as e:
        print(e)
        return
        
    speak("Hello, I'm your AI assistant. Start talking! I'll adapt to your mood.", voice)
    
    while True:
        user_text = listen(stream, recognizer)
        print("You said:", user_text)
        
        if "exit" in user_text.lower() or "quit" in user_text.lower():
            speak("Goodbye!", voice)
            break
            
        detected_mood = detect_mood_from_face()
        llm_reply = get_llm_response(user_text, detected_mood)
        speak(llm_reply, voice)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()