import os, requests, json

API_KEY = os.getenv("LLM_API_KEY")
MODEL   = "gemini-2.5-flash"   # try "gemini-1.5-flash" if this fails
URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key=AIzaSyCAi-796Ax1qeH0EpX5hmEF5FbZG_Q33yw"

payload = {
    "contents": [
        {"parts": [{"text": "Say 'Hello, this is Gemini responding properly.'"}]}
    ]
}

resp = requests.post(URL, json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
