from __future__ import annotations


CORE_GATES = (
    "sources_traceable", "license_risks_recorded", "schema_valid", "fine_semantics_valid",
    "explicit_target_leakage_zero", "exact_cross_split_zero", "normalized_cross_split_zero",
    "conflicts_audited", "privacy_processed", "reproducible", "core_tests_pass",
    "manifest_complete", "baseline_readable",
)


def evaluate_quality_gate(checks: dict[str, bool], details: dict | None = None) -> dict:
    normalized = {name: bool(checks.get(name, False)) for name in CORE_GATES}
    failed = [name for name, passed in normalized.items() if not passed]
    return {"passed": not failed, "checks": normalized, "failed_gates": failed, "details": details or {}}
