# main.py
from optimizer import TaxOptimizer
import json, os
from llm_client import LLMClient

def show_header():
    print("\n🧮 AI TAX ASSISTANT (RAG-enabled)")
    if os.path.exists("data/version_info.json"):
        with open("data/version_info.json") as f:
            info = json.load(f)
            print(f"📅 Last Updated: {info.get('last_updated')} | Source: {info.get('source')}")
def verify_gemini_connection():
    print("🔍 Verifying Gemini connection...")
    try:
        llm = LLMClient()
        ping = llm.ask("System", "ping")
        print("✅ Gemini API connected successfully.\n")
    except Exception as e:
        print("⚠️ Gemini connection failed:", e)

if __name__ == "__main__":
    show_header()
    bot = TaxOptimizer()
    print("Type 'api' to run server, 'update' to refresh dataset, or ask a tax question.")
    while True:
        query = input("\n💭 Ask (or 'exit'): ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "update":
            from update_data import update_dataset
            update_dataset()
            continue
        if query.lower() == "api":
            print("Run `python api_server.py` in another terminal to start HTTP server.")
            continue
        res = bot.handle(query)
        if isinstance(res, dict):
            print("\n[DIRECT]\n", res["DIRECT"].summary)
            print("\n[INDIRECT]\n", res["INDIRECT"].summary)
        else:
            print(f"\n[{res.category}]\n{res.summary}\n")
            print("Suggestions:")
            for s in res.suggestions:
                print(" -", s)
