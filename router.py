# router.py
from llm_client import LLMClient

class TaxRouter:
    def __init__(self):
        self.llm = LLMClient()

    def classify(self, text: str):
        txt = text.lower()
        if any(x in txt for x in ["gst", "vat", "itc", "indirect", "gstr"]):
            return "INDIRECT"
        if any(x in txt for x in ["income", "salary", "itr", "80c", "80d", "deduction", "direct"]):
            return "DIRECT"
        # fallback via LLM:
        try:
            sys = "Classify a user tax question as DIRECT, INDIRECT or BOTH. Answer with one of those words only."
            ans = self.llm.ask(sys, text)
            ans = ans.strip().upper()
            if "DIRECT" in ans:
                return "DIRECT"
            if "INDIRECT" in ans:
                return "INDIRECT"
            if "BOTH" in ans:
                return "BOTH"
        except Exception:
            pass
        return "DIRECT"
