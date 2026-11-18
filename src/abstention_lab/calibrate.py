from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+")
STOP = {
    "politica", "política", "cuales", "cuáles", "cual", "cuál", "como", "cómo",
    "que", "qué", "tiene", "tienen", "para", "sobre",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


def load_chunks(docs_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        chunks.append({"doc_id": path.stem, "text": text})
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        n = len(tokenized) or 1
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        self.vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            length = len(tokens) or 1
            self.vectors.append({t: (c / length) * self.idf[t] for t, c in tf.items()})

    def top_score(self, query: str) -> float:
        q_tokens = tokenize(query)
        tf = Counter(q_tokens)
        length = len(q_tokens) or 1
        q_vec = {t: (c / length) * self.idf.get(t, 0.0) for t, c in tf.items()}
        best = 0.0
        for vec in self.vectors:
            keys = set(q_vec) | set(vec)
            if not keys:
                continue
            dot = sum(q_vec.get(x, 0.0) * vec.get(x, 0.0) for x in keys)
            na = math.sqrt(sum(v * v for v in q_vec.values()) or 1)
            nb = math.sqrt(sum(v * v for v in vec.values()) or 1)
            best = max(best, dot / (na * nb))
        return best


def content_overlap(question: str, doc_text: str) -> float:
    content = [t for t in tokenize(question) if t not in STOP and len(t) > 3]
    if not content:
        return 0.0
    ctx = set(tokenize(doc_text))
    return sum(1 for t in content if t in ctx) / len(content)


def should_answer(retriever: Retriever, chunks: list[dict], question: str, min_score: float, min_overlap: float) -> bool:
    score = retriever.top_score(question)
    if score < min_score:
        return False
    best_doc = max(chunks, key=lambda c: content_overlap(question, c["text"]))
    return content_overlap(question, best_doc["text"]) >= min_overlap


def calibrate(docs_dir: Path, cases_path: Path) -> dict:
    chunks = load_chunks(docs_dir)
    retriever = Retriever(chunks)
    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    grid = []
    for min_score in (0.08, 0.12, 0.18, 0.25, 0.35):
        for min_overlap in (0.20, 0.34, 0.50):
            correct = 0
            for case in cases:
                predicted = should_answer(retriever, chunks, case["question"], min_score, min_overlap)
                expected = not case.get("should_abstain", False)
                correct += int(predicted == expected)
            acc = correct / len(cases)
            grid.append(
                {
                    "min_score": min_score,
                    "min_overlap": min_overlap,
                    "accuracy": round(acc, 4),
                    "correct": correct,
                    "total": len(cases),
                }
            )
    best = max(grid, key=lambda x: (x["accuracy"], x["min_score"], x["min_overlap"]))
    return {"best": best, "grid": grid, "score": best["accuracy"]}
