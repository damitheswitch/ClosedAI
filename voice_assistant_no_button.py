import json
import pyaudio
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from piper.voice import PiperVoice
from openai import OpenAI
import os
from dotenv import load_dotenv

# NEW IMPORTS FOR FACIAL MOOD DETECTION
import cv2  # For video capture from phone stream
from deepface import DeepFace  # For emotion analysis (happy, sad, etc.)

load_dotenv()

# Vosk STT setup
vosk_model = Model("vosk_model")
recognizer = KaldiRecognizer(vosk_model, 48000)

# PipeWire audio setup - INITIALIZE THIS FIRST!
MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))  # USB mic
p = pyaudio.PyAudio()

# Add retry logic for microphone
for attempt in range(5):
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, 
                       input=True, input_device_index=MIC_SOURCE_ID, 
                       frames_per_buffer=16384)
        stream.start_stream()
        print("Microphone opened successfully!")
        break
    except OSError as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < 4:
            import time
            time.sleep(1)
        else:
            print("Failed to open microphone after 5 attempts")
            exit(1)

# Piper TTS setup
piper_model_path = "piper_voice/en_US-lessac-medium.onnx"
voice = PiperVoice.load(piper_model_path)

# LLM API setup
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

# ROBUST VERSION: Detect mood from facial expression
def detect_mood_from_face():
    """
    Robust mood detection with multiple fallbacks and error handling
    """
    # First, check if the model file exists and is valid
    model_path = "/home/imad/.deepface/weights/facial_expression_model_weights.h5"
    if os.path.exists(model_path):
        try:
            # Test if the model file is valid by checking its size
            file_size = os.path.getsize(model_path)
            if file_size < 1000000:  # If file is too small, it's probably corrupted
                print("Model file appears corrupted, removing...")
                os.remove(model_path)
        except:
            pass
    
    try:
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            print(f"Trying camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"Camera {camera_index} not accessible")
                continue
                
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Give camera time to initialize
            import time
            time.sleep(0.5)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"Camera {camera_index} opened but couldn't capture frame")
                continue
                
            print(f"Successfully captured frame from camera {camera_index}")
            
            # Save the frame for debugging
            cv2.imwrite('/tmp/last_capture.jpg', frame)
            
            # Resize for better performance
            frame = cv2.resize(frame, (224, 224))
            
            # Analyze with comprehensive error handling
            try:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    silent=True,
                    detector_backend='opencv'  # Use opencv for better compatibility
                )
                
                if result and len(result) > 0:
                    mood = result[0]['dominant_emotion']
                    print(f"Success! Detected mood: {mood}")
                    return mood
                else:
                    print("DeepFace returned empty result")
                    
            except Exception as e:
                print(f"DeepFace analysis failed: {e}")
                # Check if it's a model download issue
                if "download" in str(e).lower() or "weights" in str(e).lower():
                    print("Model file issue detected, attempting cleanup...")
                    try:
                        if os.path.exists(model_path):
                            os.remove(model_path)
                            print("Removed potentially corrupted model file")
                    except:
                        pass
                continue
                
    except Exception as e:
        print(f"Camera access failed completely: {e}")
    
    # Final fallback: use a simple simulated mood based on time
    print("Using fallback mood detection")
    try:
        import datetime
        hour = datetime.datetime.now().hour
        if 6 <= hour < 12:
            return "happy"
        elif 12 <= hour < 18:
            return "neutral"
        elif 18 <= hour < 22:
            return "relaxed"
        else:
            return "sleepy"
    except:
        return "neutral"

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

def get_llm_response(user_input, mood="neutral"):
    """
    Generates LLM response, adapting system prompt based on detected mood.
    """
    # Mood-specific instructions
    mood_instructions = {
        "happy": "Respond enthusiastically and positively to match the user's happy mood.",
        "sad": "Respond empathetically and supportively to cheer up the sad user.",
        "angry": "Respond calmly and apologetically to de-escalate the angry user.",
        "neutral": "Respond normally.",
        "fear": "Respond reassuringly and comforting.",
        "surprise": "Respond with excitement and curiosity.",
        "disgust": "Respond with understanding and offer alternatives.",
        "relaxed": "Respond in a calm and peaceful manner.",
        "sleepy": "Respond gently and quietly.",
        "angry": "Respond very calmly and carefully."
    }.get(mood, "Respond normally.")
    
    system_prompt = f"You are a voice assistant. {mood_instructions} Give short, concise answers. Never use markdown, asterisks, hashes, or any formatting. Speak naturally like a human."
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content.strip()

def clean_response(text):
    # Remove markdown symbols
    text = text.replace('*', '').replace('#', '').replace('`', '')
    text = text.replace('**', '').replace('**', '').replace('__', '')
    return text

# Main loop
speak("Hello, I'm your AI assistant. Start talking! I'll adapt to your mood.")
while True:
    user_text = listen()
    print("You said:", user_text)
    if "exit" in user_text.lower() or "quit" in user_text.lower():
        speak("Goodbye!")
        break
    
    # Detect mood from face before responding
    detected_mood = detect_mood_from_face()
    
    # Get adaptive LLM response
    llm_reply = get_llm_response(user_text, detected_mood)
    cleaned_reply = clean_response(llm_reply)
    speak(cleaned_reply)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
