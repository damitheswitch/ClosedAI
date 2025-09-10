import json
import pyaudio
import vosk
import os
import sys
from typing import Optional, Callable

class STTEngine:
    """
    Speech-to-Text engine using Vosk for Raspberry Pi
    """
    
    def __init__(self, model_path: str = None, sample_rate: int = 16000):
        """
        Initialize the STT engine with Vosk model
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate (16000 is recommended for Vosk)
        """
        self.sample_rate = sample_rate
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.recognizer = None
        self.audio = None
        self.stream = None
        self.is_listening = False
        
        # Initialize the model
        self._initialize_model()
    
    def _get_default_model_path(self) -> str:
        """
        Get the default model path for Vosk
        For Raspberry Pi, we'll use a smaller model
        """
        # You can download models from: https://alphacephei.com/vosk/models
        # For Raspberry Pi, use the smaller models like:
        # - vosk-model-small-en-us-0.15
        # - vosk-model-en-us-0.22 (larger but more accurate)
        
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_name = "vosk-model-small-en-us-0.15"  # Change this to your downloaded model
        
        return os.path.join(models_dir, model_name)
    
    def _initialize_model(self):
        """
        Initialize the Vosk model and recognizer
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Vosk model not found at: {self.model_path}")
            
            print(f"Loading Vosk model from: {self.model_path}")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            print("✅ Vosk model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading Vosk model: {e}")
            print("\n📥 To download a model:")
            print("1. Go to: https://alphacephei.com/vosk/models")
            print("2. Download a model (recommended: vosk-model-small-en-us-0.15)")
            print("3. Extract it to the 'models' directory")
            sys.exit(1)
    
    def _initialize_audio(self):
        """
        Initialize PyAudio for microphone input
        """
        try:
            self.audio = pyaudio.PyAudio()
            
            # Find the default microphone
            device_index = self._find_microphone()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=4000
            )
            
            print("✅ Microphone initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing microphone: {e}")
            raise
    
    def _find_microphone(self) -> Optional[int]:
        """
        Find the default microphone device
        """
        try:
            info = self.audio.get_default_input_device_info()
            return info['index']
        except:
            # Fallback to device 0
            return 0
    
    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for speech once and return the recognized text
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if no speech detected
        """
        if not self.stream:
            self._initialize_audio()
        
        print("🎤 Listening... (speak now)")
        
        try:
            # Read audio data
            data = self.stream.read(4000, exception_on_overflow=False)
            
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip()
                
                if text:
                    print(f"🎯 Recognized: {text}")
                    return text
                else:
                    print("�� No speech detected")
                    return None
            else:
                # Partial result
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                if partial_text:
                    print(f"⏳ Partial: {partial_text}", end='\r')
                
                return None
                
        except Exception as e:
            print(f"❌ Error during speech recognition: {e}")
            return None
    
    def listen_continuous(self, callback: Callable[[str], None], stop_phrase: str = "stop listening"):
        """
        Continuously listen for speech and call callback with recognized text
        
        Args:
            callback: Function to call with recognized text
            stop_phrase: Phrase to stop continuous listening
        """
        if not self.stream:
            self._initialize_audio()
        
        self.is_listening = True
        print(f"🎤 Continuous listening started. Say '{stop_phrase}' to stop.")
        
        try:
            while self.is_listening:
                data = self.stream.read(4000, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text:
                        print(f"🎯 Recognized: {text}")
                        
                        # Check for stop phrase
                        if stop_phrase.lower() in text.lower():
                            print("🛑 Stop phrase detected. Stopping continuous listening.")
                            self.is_listening = False
                            break
                        
                        # Call the callback function
                        callback(text)
                
                else:
                    # Show partial results
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get('partial', '').strip()
                    if partial_text:
                        print(f"⏳ Partial: {partial_text}", end='\r')
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.is_listening = False
        
        except Exception as e:
            print(f"❌ Error during continuous listening: {e}")
            self.is_listening = False
    
    def stop_listening(self):
        """
        Stop continuous listening
        """
        self.is_listening = False
        print("🛑 Stopping continuous listening...")
    
    def cleanup(self):
        """
        Clean up resources
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("�� STT engine cleaned up")

# Convenience functions for easy usage
def listen_once(model_path: str = None) -> Optional[str]:
    """
    Simple function to listen once and return text
    """
    stt = STTEngine(model_path)
    try:
        return stt.listen_once()
    finally:
        stt.cleanup()

def listen_continuous(callback: Callable[[str], None], model_path: str = None, stop_phrase: str = "stop listening"):
    """
    Simple function for continuous listening
    """
    stt = STTEngine(model_path)
    try:
        stt.listen_continuous(callback, stop_phrase)
    finally:
        stt.cleanup()

# Example usage
if __name__ == "__main__":
    def on_speech_detected(text: str):
        print(f"📝 Processing: {text}")
        # Here you would typically send the text to your NLP engine
    
    # Test the STT engine
    print("�� Testing STT Engine...")
    
    # Single recognition
    print("\n1. Single recognition test:")
    text = listen_once()
    if text:
        print(f"✅ You said: {text}")
    else:
        print("❌ No speech detected")
    
    # Continuous recognition
    print("\n2. Continuous recognition test:")
    print("Say 'stop listening' to end the test")
    listen_continuous(on_speech_detected)
