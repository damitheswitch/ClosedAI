# ClosedAI Mood Detection Assistant (RPI Online)

A voice-activated AI assistant that adapts responses based on mood detection, running on a Raspberry Pi with video streaming from a phone.

## Team
- Lamini Mohamed (laminimohamed)
- Mohamed Amine Ait El Mahjoub (Amineeait)
- Imad Charradi (damitheswitch)

## Features
- Real-time speech recognition using Vosk.
- Mood detection from facial expressions via DeepFace and phone camera stream.
- Adaptive LLM responses based on detected mood (via OpenAI).
- Streams video from DroidCam on Android to Pi using v4l2loopback.

## Installation

### Prerequisites
- Raspberry Pi (tested on Pi 4 with Bookworm).
- Android phone with DroidCam app.
- Internet connection (for OpenAI API).

### Steps
1. **Set Up Raspberry Pi**:
   - Update system: `sudo apt update && sudo apt upgrade`.
   - Install dependencies: `sudo apt install ffmpeg v4l2loopback-dkms v4l2loopback-utils python3-pip python3-venv`.
   - Clone repo: `git clone https://github.com/damitheswitch/ClosedAI.git; cd ClosedAI/Mood_detection_feature-RPI_online`.
   - Create virtual env: `python3 -m venv .venv; source .venv/bin/activate`.

2. **Install Python Packages**:
   - `pip install -r requirements.txt`.

3. **Configure Video Streaming**:
   - Load v4l2loopback: `sudo modprobe v4l2loopback devices=1 video_nr=0 card_label="DroidCam" exclusive_caps=1 keep_format=1`.
   - Start DroidCam on phone (WiFi mode, IP e.g., 192.168.125.251:4747).
   - Stream: `ffmpeg -f mjpeg -i http://192.168.125.251:4747/video -f v4l2 -pix_fmt yuv420p -framerate 10 /dev/video0 &`.

4. **Run the Assistant**:
   - Set OpenAI API key in `.env` (e.g., `OPENAI_API_KEY=your_key`).
   - Run: `python voice_assistant_no_button.py`.
   - Speak to mic, face phone camera—say "exit" to quit.

## Usage
- Start talking after the greeting.
- Mood adapts responses (e.g., cheerful for "happy").
- Stream stops with Ctrl+C in ffmpeg terminal.

## Testing
- Run unit tests: `python run_tests.py`.

## Known Issues
- Mood detection may be inaccurate with poor lighting or angles.
- Streaming fails if DroidCam app closes—restart app.

## License
This is for a school project in September 2025. Feel free to use this repository however you like.
