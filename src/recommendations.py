from __future__ import annotations

import re


def rule_based_recommendations(
    resume_text: str,
    missing_skills: list[str],
    completeness_missing: list[str],
) -> list[str]:
    tips: list[str] = []

    if missing_skills:
        preview = ", ".join(missing_skills[:6])
        tips.append(
            f"Prioritize genuine evidence for relevant missing skills: {preview}. "
            "Do not add skills you cannot explain in an interview."
        )

    # Approximate evidence of metrics in project/experience bullets.
    metric_hits = re.findall(
        r"(?:\b\d+(?:\.\d+)?%|\b\d+x\b|\b\d+\+?\s+(?:users|records|samples|requests|documents|images)\b)",
        resume_text,
        flags=re.IGNORECASE,
    )
    if not metric_hits:
        tips.append(
            "Quantify project or experience bullets where possible: dataset size, "
            "latency, accuracy, error reduction, users, throughput, or other measurable outcomes."
        )

    if not re.search(r"github\.com|github", resume_text, re.I):
        tips.append(
            "Add a GitHub/project link for technical projects if you have a clean public repository."
        )

    word_count = len(resume_text.split())
    if word_count > 900:
        tips.append(
            "The resume is text-heavy. Consider compressing weaker bullets and keeping the most role-relevant evidence."
        )
    elif word_count < 220:
        tips.append(
            "The resume is quite short. Add concrete project/experience evidence rather than generic objective statements."
        )

    for missing in completeness_missing:
        tips.append(f"Resume structure check: consider adding or clarifying the {missing.lower()}.")

    return tips[:7]


def rule_based_interview_questions(
    matched_skills: list[str],
    missing_skills: list[str],
) -> list[str]:
    questions = [
        "Walk me through one project end-to-end: problem, data/input, approach, trade-offs, evaluation, and deployment.",
        "What was the hardest technical decision in your strongest project, and what alternatives did you consider?",
        "Describe one failure or poor result you encountered and how you debugged it.",
    ]

    for skill in matched_skills[:4]:
        questions.append(
            f"Your resume and this role both mention {skill}. Explain where you used it and one technical limitation or trade-off."
        )

    for skill in missing_skills[:3]:
        questions.append(
            f"The role mentions {skill}. What do you know about it, and how would you learn or apply it for this job?"
        )

    return questions[:10]
