import requests

# 🔑 DeepSeek API Key (replace with env var in production)
API_KEY = "sk-c36ea8e4c694498c9bdfd98ac3350d7e"
API_URL = "https://api.deepseek.com/v1/chat/completions"

def respond(user_text: str) -> str:
    """
    Sends user_text to DeepSeek API and returns the assistant's reply.
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",   # change if your model is different
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # throw error if status not 200

        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return reply

    except requests.exceptions.RequestException as e:
        print("❌ Error calling DeepSeek API:", e)
        return "Sorry, I couldn’t connect to DeepSeek right now."
