# analyzers/base_analyzer.py
from abc import ABC, abstractmethod

class TaxAnalyzer(ABC):
    @abstractmethod
    def analyze(self, query, memory_context=""):
        raise NotImplementedError
