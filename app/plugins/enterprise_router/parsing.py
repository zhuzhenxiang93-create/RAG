"""Robust but strict parsing of the router model's JSON output."""

import json
from typing import Tuple

from pydantic import ValidationError

from app.plugins.enterprise_router.schema import RouteDecision


def _extract_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("router output does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("router output contains an incomplete JSON object")


def parse_route(text: str) -> Tuple[RouteDecision, str]:
    """Return a validated route and its canonical JSON representation."""
    try:
        payload = json.loads(_extract_object(text))
        decision = RouteDecision.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("router output does not match RouteDecision") from exc
    return decision, decision.canonical_json()

