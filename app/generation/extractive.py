"""Evidence-only answer generation for Lite mode."""

import re
from typing import List, Set

from app.retrieval.tokenizer import tokenize
from app.schemas.qa import ConflictItem, EvidenceItem


class ExtractiveGenerator:
    """Select source sentences and never add unsupported domain facts."""

    _sentence_boundary = re.compile(r"(?<=[。！？.!?])\s+|[\r\n]+")

    def generate(
        self,
        query: str,
        evidence: List[EvidenceItem],
        conflicts: List[ConflictItem],
    ) -> str:
        if conflicts:
            sources = sorted(
                {
                    item.filename
                    for item in evidence
                    if item.relation == "contradict"
                }
            )
            return "检索到相互冲突的证据（{}），暂不生成单一结论，请确认适用版本或时间范围。".format(
                "、".join(sources)
            )

        supporting = [item for item in evidence if item.relation == "support"]
        if not supporting:
            return "当前文档中没有找到足以支持回答的证据。"

        query_tokens = self._tokens(query)
        selected: List[str] = []
        for item in supporting[:3]:
            sentence = self._best_sentence(item.text, query_tokens)
            if sentence and sentence not in selected:
                selected.append("{} [{}]".format(sentence, item.evidence_id))
            if len(selected) == 3:
                break
        return " ".join(selected) if selected else "当前证据不足以形成可靠答案。"

    def _best_sentence(self, text: str, query_tokens: Set[str]) -> str:
        sentences = [
            sentence.strip()
            for sentence in self._sentence_boundary.split(text)
            if sentence.strip()
        ]
        if not sentences:
            return ""
        ranked = sorted(
            sentences,
            key=lambda sentence: (
                -len(query_tokens & self._tokens(sentence)),
                len(sentence),
            ),
        )
        best = ranked[0]
        return best if len(best) <= 350 else best[:349] + "…"

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {token for token in tokenize(text) if len(token) >= 2}
