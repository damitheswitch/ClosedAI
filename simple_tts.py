import pyttsx3
import threading

class SimpleTTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Set voice properties
        self.engine.setProperty('rate', 150)    # Speaking speed
        self.engine.setProperty('volume', 0.9)  # Volume level (0-1)
        
    def speak(self, text):
        """Speak text in a separate thread to avoid blocking"""
        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
        
    def stop(self):
        """Stop any ongoing speech"""
        self.engine.stop()

# Create a global instance
tts = SimpleTTS()
