import os
import requests
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")  # 'openai' or 'gemini'
        self.key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4-turbo")
        self.endpoint = os.getenv("LLM_ENDPOINT", "")

    def ask(self, system_prompt, user_prompt, max_tokens=800):
        if self.provider == "openai":
            return self._ask_openai(system_prompt, user_prompt, max_tokens)
        elif self.provider == "gemini":
            return self._ask_gemini(system_prompt, user_prompt)
        else:
            raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'gemini'.")

    # ---------------------- OPENAI ----------------------
    def _ask_openai(self, system_prompt, user_prompt, max_tokens):
        url = self.endpoint or "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    # ---------------------- GEMINI ----------------------
        # ---------------------- GEMINI ----------------------

    def _ask_gemini(self, system_prompt, user_prompt):
        
        import json, time, requests

        def make_payload(prompt_text):
            return {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 180,
                    "topP": 0.8
            }
        }

        # Prepare prompt
        concise_prompt = (
            "You are an Indian Tax Advisor. Respond in concise bullet points.\n"
            "Include: (1) Summary, (2) 3 short tax-saving tips, "
            "(3) 💰 Estimated savings range if applicable. Keep it under 80 words.\n\n"
            f"Question: {user_prompt}"
        )

        headers = {"Content-Type": "application/json"}

        models_to_try = [self.model, "gemini-1.5-flash"]
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.key}"
            for attempt in range(3):
                try:
                    r = requests.post(url, headers=headers, json=make_payload(concise_prompt), timeout=30)
                    if r.status_code == 403:
                        print("❌ Invalid or restricted API key.")
                        break
                    if r.status_code == 429:
                        print("⚠️ Rate limit reached. Waiting 5 seconds...")
                        time.sleep(5)
                        continue
                    r.raise_for_status()
                    data = r.json()
                # Extract text safely
                    text = None
                    if "candidates" in data:
                        for cand in data["candidates"]:
                            parts = cand.get("content", {}).get("parts", [])
                            for p in parts:
                                if "text" in p:
                                    text = p["text"].strip()
                                    break
                            if text:
                                break
                    if text:
                        if len(text.split()) > 120:
                            text = " ".join(text.split()[:120]) + "..."
                        print(f"✅ Gemini ({model_name}) generated reply.")
                        return text
                    if data.get("candidates", [{}])[0].get("finishReason") == "MAX_TOKENS":
                        print("⚠️ Gemini hit token limit, reducing prompt further...")
                        concise_prompt = concise_prompt[:400]
                        time.sleep(1)
                        continue
                    print("⚠️ Unexpected Gemini response:", json.dumps(data, indent=2)[:400])
                    break
                except Exception as e:
                    print(f"❌ Gemini ({model_name}) failed attempt {attempt+1}: {e}")
                    time.sleep(1)
        return "⚠️ Gemini failed to respond after multiple attempts."





