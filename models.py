import re

class TaxQuery:
    def __init__(self, text: str, user_id: str = "anon"):
        self.text = text
        self.user_id = user_id

class TaxAnswer:
    def __init__(self, category: str, summary: str, suggestions: list, context: dict = None):
        self.category = category
        self.summary = summary
        self.suggestions = suggestions
        self.context = context or {}

    def _extract_estimated(self):
        """
        Extract approximate INR amounts or ranges from summary text.
        Returns a string like '₹1.5L-3L' or None.
        """
        text = self.summary or ""
        # look for currency symbol ₹ or INR or words like 'Estimated Tax Saving'
        # capture patterns like ₹1,50,000 or ₹1.5L or 1.5 lakh
        # normalize matches to the original substring
        m = re.search(r"(?:Estimated Tax Saving[:\s]*|Estimated Savings[:\s]*|💰\s*Estimated Tax Saving[:\s]*|\b)([₹INR\s0-9,\.\-lLakhs]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # fallback to any ₹ amount
        m2 = re.search(r"(₹[0-9,\.]+(?:\s*-\s*₹[0-9,\.]+)?)", text)
        if m2:
            return m2.group(1)
        return None

    def to_dict(self):
        est = self._extract_estimated()
        return {
            "category": self.category,
            "summary": self.summary,
            "estimated_saving": est,
            "suggestions": self.suggestions,
            "context": self.context
        }
