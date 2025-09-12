import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"Device {i}: {dev['name']}, Input Channels: {dev['maxInputChannels']}, Output Channels: {dev['maxOutputChannels']}")
p.terminate()
