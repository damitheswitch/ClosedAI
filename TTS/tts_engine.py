import os

def speak(text: str):
    """
    Convert text to speech using espeak (lightweight, works on Raspberry Pi).
    """
    if not text:
        return
    
    # Escape quotes in case text contains them
    safe_text = text.replace('"', '\\"')
    
    # Call espeak from command line
    os.system(f'espeak "{safe_text}" 2>/dev/null')
