from gtts import gTTS
import os

# Text you want to convert to speech
text_to_speak = "Hello, this is a test of Google's text to speech."

try:
    # Create a gTTS object
    tts = gTTS(text=text_to_speak, lang='en')

    # Save the audio to a temporary file
    temp_file = "temp_audio.mp3"
    tts.save(temp_file)
    print(f"Audio saved to {temp_file}")

    # Play the audio file using mpg123
    print("Playing audio...")
    os.system(f"mpg123 {temp_file}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print("Temporary file deleted.")
