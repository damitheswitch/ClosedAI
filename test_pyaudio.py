import pyaudio
import wave
import os

MIC_SOURCE_ID = int(os.getenv("MIC_SOURCE_ID", "1"))
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, input_device_index=MIC_SOURCE_ID, frames_per_buffer=16384)
stream.start_stream()

print("Recording... (5 seconds, then stop)")
frames = []
for _ in range(int(48000 / 4096 * 5)):
    data = stream.read(4096, exception_on_overflow=False)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("test_pyaudio.wav", "wb")
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(48000)
wf.writeframes(b"".join(frames))
wf.close()

print("Saved to test_pyaudio.wav")
