import json
import os
import requests
import numpy as np
import faiss
import hashlib
from datetime import datetime

class VectorMemory:
    def __init__(self, memory_file="data/memory_store.json", dim=768):
        self.memory_file = memory_file
        self.index_file = "data/faiss_index.bin"
        self.dim = dim
        self.data = self._load_memory()
        self.index = self._load_index()
        print(f"🧠 Vector memory initialized with {len(self.data)} records")

    # ------------------------- #
    # File handling
    # ------------------------- #
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _load_index(self):
        if os.path.exists(self.index_file):
            return faiss.read_index(self.index_file)
        return faiss.IndexFlatL2(self.dim)

    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)

    # ------------------------- #
    # Embedding Logic (Gemini)
    # ------------------------- #
    def _hash(self, text):
        """Hash the text to check duplicates quickly."""
        return hashlib.sha256(text.lower().encode()).hexdigest()

    def _embed(self, text):
        """Get vector embedding from cache or Gemini."""
        # 1️⃣ Check if this query already exists in memory
        text_hash = self._hash(text)
        for item in self.data:
            if item.get("hash") == text_hash:
                print(f"⚡ Cached embedding used for: {text[:40]}...")
                return np.array(item["vector"], dtype=np.float32)

        # 2️⃣ Otherwise, call Gemini Embedding API
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("Missing LLM_API_KEY environment variable")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
        body = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}

        try:
            r = requests.post(url, json=body, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "embedding" not in data:
                raise ValueError("Unexpected embedding format: " + json.dumps(data))
            vec = np.array(data["embedding"]["values"], dtype=np.float32)
            print(f"✅ New embedding fetched for: {text[:40]}...")
            return vec
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return np.zeros(self.dim, dtype=np.float32)

    # ------------------------- #
    # Add + Retrieve Memory
    # ------------------------- #
    def add(self, query, answer):
        """Add new query-answer pair to memory."""
        vector = self._embed(query)
        query_hash = self._hash(query)

        self.index.add(np.array([vector]))
        self.data.append({
            "hash": query_hash,
            "query": query,
            "answer": answer,
            "vector": vector.tolist(),
            "timestamp": datetime.now().isoformat()
        })

        self._save_memory()
        self._save_index()
        print(f"💾 Memory updated with new entry: {query[:50]}...")

    def retrieve(self, query, top_k=3):
        """Retrieve top_k most relevant past answers."""
        if not self.data:
            return []

        vector = self._embed(query)
        D, I = self.index.search(np.array([vector]), top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.data):
                results.append(self.data[idx])
        print(f"🔍 Retrieved {len(results)} similar results for query.")
        return results
