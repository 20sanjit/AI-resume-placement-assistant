from __future__ import annotations

import json
import re
from pathlib import Path


class SkillExtractor:
    def __init__(self, skills_path: str | Path):
        self.skills_path = Path(skills_path)
        with self.skills_path.open("r", encoding="utf-8") as f:
            self.skill_map: dict[str, list[str]] = json.load(f)

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        # Non-word lookarounds work better than \b for terms like C++ / .NET.
        pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    def extract(self, text: str) -> set[str]:
        found: set[str] = set()
        for canonical, aliases in self.skill_map.items():
            terms = [canonical, *aliases]
            if any(self._contains(text, term) for term in terms):
                found.add(canonical)
        return found

    @staticmethod
    def compare(
        resume_skills: set[str],
        jd_skills: set[str],
    ) -> dict[str, object]:
        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)

        if jd_skills:
            coverage = 100.0 * len(matched) / len(jd_skills)
        else:
            coverage = 0.0

        return {
            "matched": matched,
            "missing": missing,
            "resume_only": sorted(resume_skills - jd_skills),
            "coverage": coverage,
        }
