# optimizer.py
from router import TaxRouter
from analyzers.direct_analyzer import DirectTaxAnalyzer
from analyzers.indirect_analyzer import IndirectTaxAnalyzer
from memory.vector_memory import VectorMemory
from models import TaxQuery

class TaxOptimizer:
    def __init__(self):
        self.router = TaxRouter()
        self.direct = DirectTaxAnalyzer()
        self.indirect = IndirectTaxAnalyzer()
        self.memory = VectorMemory()

    def handle(self, query_text):
        q = TaxQuery(query_text)
        # retrieve similar past Q/A
        past = self.memory.retrieve(query_text, top_k=3)
        context = "\n\n".join([f"Q: {p['query']}\nA: {p['answer']}" for p in past]) if past else ""
        category = self.router.classify(query_text)
        if category == "DIRECT":
            res = self.direct.analyze(q, memory_context=context)
        elif category == "INDIRECT":
            res = self.indirect.analyze(q, memory_context=context)
        else:
            direct = self.direct.analyze(q, memory_context=context)
            indirect = self.indirect.analyze(q, memory_context=context)
            res = {"DIRECT": direct, "INDIRECT": indirect}
        # store in memory (store summary to keep memory short)
        summary = res.summary if not isinstance(res, dict) else res["DIRECT"].summary
        self.memory.add(query_text, summary)
        return res
