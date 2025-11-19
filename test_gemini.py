# test_gemini.py
from llm_client import LLMClient

def test_gemini_connection():
    print("🔍 Testing Gemini API connection...\n")
    try:
        llm = LLMClient()
        answer = llm.ask(
            "You are a helpful Indian tax assistant.",
            "What is the current income tax exemption limit in India?"
        )
        print("✅ Gemini API connection successful!\n")
        print("💬 Model response:\n")
        print(answer)
    except Exception as e:
        print("❌ Gemini API test failed.")
        print("Error details:", e)

if __name__ == "__main__":
    test_gemini_connection()
