# api_server.py
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from optimizer import TaxOptimizer
from llm_client import LLMClient 
import json, os
from fastapi.middleware.cors import CORSMiddleware
import re
from agents.portfolio_agent import optimize_portfolio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path





app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent   # tax_assistant/
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
bot = TaxOptimizer()

class QueryIn(BaseModel):
    query: str
    
@app.get("/status")
async def status():
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    provider = os.getenv("LLM_PROVIDER", "gemini")
    try:
        llm = LLMClient()
        test = llm.ask("System", "ping")
        return {
            "status": "ok",
            "provider": provider.capitalize(),
            "model": model,
            "ping": "✅ Gemini connected"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/portfolio_optimize")
def portfolio_api(payload: dict):
    result = optimize_portfolio(payload)
    return {"status": "ok", "result": result}

@app.post("/portfolio_optimize")
def portfolio_api(payload: dict):
    return optimize_portfolio(payload)

@app.post("/query")
def query_api(payload: QueryIn):
    import requests
    
    n8n_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/anas")
    
    print(f"🔧 Attempting to connect to n8n at: {n8n_url}")
    
    try:
        # First, test if n8n is reachable
        test_response = requests.get("http://localhost:5678", timeout=10)
        print(f"✅ n8n interface is reachable (status: {test_response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ n8n is not running or not accessible on localhost:5678")
        return {
            "error": "n8n service not running", 
            "details": "Please start n8n first on localhost:5678",
            "fallback_response": "I can provide basic tax advice. For property purchase tax savings, consider: Section 80C benefits for home loans, Section 24(b) for interest deductions, and Section 80EE for first-time home buyers."
        }
    
    try:
        # Try the webhook
        response = requests.post(n8n_url, json={"query": payload.query}, timeout=30)
        print(f"📡 n8n webhook response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ n8n Agent Response:", data)
            return data
        else:
            print(f"❌ n8n returned status {response.status_code}: {response.text}")
            return {
                "error": f"n8n webhook error (status {response.status_code})",
                "details": "Webhook might be misconfigured",
                "fallback_response": "While I set up the advanced tax advisor, here are some property tax tips: You can claim deductions under Section 80C for principal repayment, Section 24 for interest paid, and consider joint ownership for better tax planning."
            }
            
    except requests.exceptions.Timeout:
        print("❌ n8n request timed out")
        return {"error": "n8n timeout", "details": "n8n took too long to respond"}
    except Exception as e:
        print(f"❌ Failed to contact n8n Agent: {e}")
        return {
            "error": "n8n agent not reachable", 
            "details": str(e),
            "fallback_response": "For property tax savings, explore home loan interest deductions (Section 24), principal repayment benefits (Section 80C), and additional deductions for first-time home buyers."
        }

@app.post("/query")
def query_api(payload: QueryIn):
    query_text = payload.query

    # 1️⃣ Extract stock tickers (e.g., INFY.NS, TCS.NS)
    tickers = re.findall(r"\b[A-Z]{2,6}\.(NS|BO)\b", query_text)

    # If tickers were found → run portfolio optimizer
    if tickers:
        print("📊 Extracted tickers:", tickers)
        result = optimize_portfolio({"tickers": tickers})
        return {
            "mode": "portfolio",
            "tickers": tickers,
            "result": result
        }

    # Otherwise → send to n8n (your tax agent)
    import requests
    n8n_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/tax-agent")

    try:
        response = requests.post(n8n_url, json={"query": payload.query}, timeout=60)
        response.raise_for_status()
        return {
            "mode": "tax_advice",
            "result": response.json()
        }
    except Exception as e:
        return {"error": "n8n agent unreachable", "details": str(e)}
@app.post("/update_dataset")
def update_dataset_api():
    from update_data import update_dataset
    ok = update_dataset()
    return {"ok": ok}

@app.get("/version")
def version():
    if os.path.exists("data/version_info.json"):
        with open("data/version_info.json") as f:
            return json.load(f)
    return {"message": "no version info"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", 8080)))







