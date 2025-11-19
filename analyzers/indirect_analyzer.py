# analyzers/indirect_analyzer.py
import json
from llm_client import LLMClient
from models import TaxAnswer

class IndirectTaxAnalyzer:
    def __init__(self):
        self.llm = LLMClient()
        with open("data/gst_rates.json") as f:
            self.gst = json.load(f)

    def analyze(self, query, memory_context=""):
        system_prompt = "You are a GST expert. Provide concise factual guidance and 3 compliance suggestions."
        facts = f"GST rates: {self.gst}"
        user_prompt = f"User query: {query.text}\n\nPast related Q/A:\n{memory_context}\n\nFacts:\n{facts}\n\nExplain briefly and give 3 practical compliance/action steps."
        summary = self.llm.ask(system_prompt, user_prompt)
        suggestions = [
            "Ensure supplier invoices include GSTIN before claiming ITC.",
            "Reconcile purchases and sales before filing GSTR-3B.",
            "Classify supplies correctly (HSN/SAC codes)."
        ]
        return TaxAnswer("INDIRECT", summary, suggestions, {"facts_used": ["gst_rates"]})
