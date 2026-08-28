from __future__ import annotations

from legalmind.schemas import SearchHit


def build_grounded_prompt(query: str, hits: list[SearchHit]) -> str:
    evidence = "\n\n".join(
        f"[{hit.chunk_id}]\n罪名标签：{','.join(hit.accusations) or '未知'}\n"
        f"相关法条：{','.join(map(str, hit.relevant_articles)) or '未知'}\n案情：{hit.text}"
        for hit in hits
    )
    return f"""你是法律案例检索助手。仅依据证据回答，并在每个关键结论后引用 chunk_id。
无法从证据确认时明确写“证据不足”。输出 JSON，字段为：
predicted_charges、analysis、similar_cases、uncertainties。

用户案情：
{query}

检索证据：
{evidence}
"""
