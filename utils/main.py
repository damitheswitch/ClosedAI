#!/usr/bin/env python3
"""
miHelpa - Voice Assistant Main Application
Orchestrates STT, NLP, and TTS modules for a complete voice assistant experience
"""

import sys
import os
import time
import signal
from typing import Optional

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from STT.stt_engine import STTEngine
from NLP.nlp_engine import respond
from TTS.tts_engine import speak

class VoiceAssistant:
    """
    Main voice assistant class that orchestrates all components
    """
    
    def __init__(self):
        self.stt_engine = None
        self.is_running = False
        self.wake_word = "hey mihelpa"  # Wake word to activate assistant
        self.exit_phrases = ["goodbye", "exit", "quit", "stop", "bye"]
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully
        """
        print("\n🛑 Shutdown signal received...")
        self.stop()
    
    def initialize(self):
        """
        Initialize all components
        """
        print("�� Initializing miHelpa Voice Assistant...")
        
        try:
            # Initialize STT engine
            print("📡 Initializing Speech-to-Text...")
            self.stt_engine = STTEngine()
            print("✅ STT Engine ready!")
            
            # Test TTS
            print("🔊 Testing Text-to-Speech...")
            speak("miHelpa is ready to help!")
            print("✅ TTS Engine ready!")
            
            print("🎉 All systems initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def process_command(self, text: str) -> bool:
        """
        Process a voice command through the NLP pipeline
        
        Args:
            text: Recognized speech text
            
        Returns:
            True if should continue, False if should exit
        """
        if not text:
            return True
        
        # Check for exit phrases
        if any(phrase in text.lower() for phrase in self.exit_phrases):
            print("👋 Exit command detected")
            speak("Goodbye! Have a great day!")
            return False
        
        # Check for wake word (for continuous mode)
        if self.wake_word in text.lower():
            print(f"�� Wake word '{self.wake_word}' detected!")
            speak("Yes, I'm listening!")
            return True
        
        # Process the command through NLP
        print(f"🧠 Processing: {text}")
        try:
            response = respond(text)
            print(f"🤖 Response: {response}")
            
            # Speak the response
            speak(response)
            
        except Exception as e:
            error_msg = "Sorry, I encountered an error processing your request."
            print(f"❌ NLP Error: {e}")
            speak(error_msg)
        
        return True
    
    def run_single_mode(self):
        """
        Run in single command mode - listen once, process, repeat
        """
        print("\n🎤 Single Command Mode")
        print("Say something and I'll respond. Say 'goodbye' to exit.")
        print("-" * 50)
        
        while self.is_running:
            try:
                # Listen for a single command
                text = self.stt_engine.listen_once()
                
                if text:
                    # Process the command
                    if not self.process_command(text):
                        break
                else:
                    print("⏳ No speech detected, listening again...")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in single mode: {e}")
                time.sleep(2)
    
    def run_continuous_mode(self):
        """
        Run in continuous mode - always listening for wake word
        """
        print("\n�� Continuous Mode")
        print(f"Always listening for '{self.wake_word}' wake word")
        print("Say 'goodbye' to exit.")
        print("-" * 50)
        
        def on_speech_detected(text: str):
            """Callback for continuous speech detection"""
            if not self.process_command(text):
                self.stt_engine.stop_listening()
        
        try:
            # Start continuous listening
            self.stt_engine.listen_continuous(
                callback=on_speech_detected,
                stop_phrase="goodbye"
            )
        except Exception as e:
            print(f"❌ Error in continuous mode: {e}")
    
    def run_interactive_mode(self):
        """
        Run in interactive mode - menu-driven interface
        """
        while self.is_running:
            print("\n" + "="*50)
            print("🎛️  miHelpa Voice Assistant - Interactive Mode")
            print("="*50)
            print("1. Single Command Mode")
            print("2. Continuous Mode (Wake Word)")
            print("3. Test TTS")
            print("4. Test STT")
            print("5. Exit")
            print("-" * 50)
            
            try:
                choice = input("Choose an option (1-5): ").strip()
                
                if choice == "1":
                    self.run_single_mode()
                elif choice == "2":
                    self.run_continuous_mode()
                elif choice == "3":
                    test_text = input("Enter text to speak: ").strip()
                    if test_text:
                        speak(test_text)
                elif choice == "4":
                    print("🎤 Say something...")
                    text = self.stt_engine.listen_once()
                    if text:
                        print(f"✅ You said: {text}")
                    else:
                        print("❌ No speech detected")
                elif choice == "5":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def start(self, mode: str = "interactive"):
        """
        Start the voice assistant
        
        Args:
            mode: 'single', 'continuous', or 'interactive'
        """
        if not self.initialize():
            return
        
        self.is_running = True
        
        try:
            if mode == "single":
                self.run_single_mode()
            elif mode == "continuous":
                self.run_continuous_mode()
            else:  # interactive
                self.run_interactive_mode()
                
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop the voice assistant and cleanup resources
        """
        self.is_running = False
        
        if self.stt_engine:
            self.stt_engine.cleanup()
        
        print("🛑 miHelpa Voice Assistant stopped")

def main():
    """
    Main entry point
    """
    print("🎯 miHelpa Voice Assistant")
    print("Manufacturing Practice Project")
    print("=" * 40)
    
    # Parse command line arguments
    mode = "interactive"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["single", "continuous", "interactive"]:
            print("❌ Invalid mode. Use: single, continuous, or interactive")
            sys.exit(1)
    
    # Create and start the assistant
    assistant = VoiceAssistant()
    
    try:
        assistant.start(mode)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

