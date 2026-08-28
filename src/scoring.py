from __future__ import annotations


def overall_match_score(
    semantic_score: float,
    skill_score: float,
    completeness_score: float,
) -> float:
    """Transparent project heuristic, not an employer ATS score."""
    score = (
        0.50 * semantic_score
        + 0.35 * skill_score
        + 0.15 * completeness_score
    )
    return round(max(0.0, min(100.0, score)), 1)


def score_label(score: float) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 65:
        return "Good match"
    if score >= 50:
        return "Moderate match"
    return "Needs improvement"
