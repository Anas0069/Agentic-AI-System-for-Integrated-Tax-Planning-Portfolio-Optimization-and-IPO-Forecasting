# analyzers/direct_analyzer.py
import json
from llm_client import LLMClient
from models import TaxAnswer

class DirectTaxAnalyzer:
    def __init__(self):
        self.llm = LLMClient()
        with open("data/deductions.json") as f:
            self.deductions = json.load(f)
        with open("data/tax_slabs.json") as f:
            self.slabs = json.load(f)

    def analyze(self, query, memory_context=""):
        system_prompt = "You are an expert in Indian income tax (direct taxes). Provide concise, factual answers and 3 actionable suggestions."
        facts = f"Tax slabs: {self.slabs}\nDeductions summary: {self.deductions}"
        user_prompt = f"User query: {query.text}\n\nPast related Q/A:\n{memory_context}\n\nFacts:\n{facts}\n\nAnswer briefly and give 3 actionable tax-saving suggestions."
        summary = self.llm.ask(system_prompt, user_prompt)
        suggestions = [
            "Consider 80C investments (PPF/ELSS/EPF) to save up to ₹1.5L.",
            "Claim 80D for medical insurance premium (limits apply).",
            "Assess old vs new tax regime by projecting taxable income."
        ]
        return TaxAnswer("DIRECT", summary, suggestions, {"facts_used": ["tax_slabs", "deductions"]})
