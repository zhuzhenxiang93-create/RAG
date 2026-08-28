from __future__ import annotations

from typing import Protocol

from legalmind.generation.prompts import build_grounded_prompt
from legalmind.models.inference import ChargeClassifier
from legalmind.retrieval.retriever import HybridRetriever


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class LegalMindPipeline:
    def __init__(
        self,
        classifier: ChargeClassifier,
        retriever: HybridRetriever,
        generator: TextGenerator | None = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator

    def analyze(self, fact: str, top_k: int = 5) -> dict:
        classification = self.classifier.predict(fact)
        labels = {item.label for item in classification.labels}
        hits = self.retriever.search(fact, predicted_labels=labels, final_k=top_k)
        result = {
            "classification": classification.model_dump(),
            "evidence": [hit.model_dump() for hit in hits],
        }
        if self.generator:
            result["report"] = self.generator.generate(build_grounded_prompt(fact, hits))
        return result
