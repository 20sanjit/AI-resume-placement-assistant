from __future__ import annotations

import re
import unicodedata


SECTION_ALIASES = {
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "internships", "internship"
    ],
    "projects": [
        "projects", "academic projects", "personal projects",
        "selected projects", "project experience"
    ],
    "skills": [
        "skills", "technical skills", "technologies", "technical expertise"
    ],
    "education": [
        "education", "academic background", "academics", "qualifications"
    ],
    "achievements": [
        "achievements", "awards", "honors", "positions of responsibility"
    ],
}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def redact_contact_info(text: str) -> str:
    """Remove common personal contact fields before sending context to an LLM."""
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    # Loose India/international-style phone matcher, deliberately conservative.
    text = re.sub(
        r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\d[-.\s]?){9,12}(?!\d)",
        "[PHONE REDACTED]",
        text,
    )
    return text


def _heading_to_key(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
    for key, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return key
    return None


def extract_sections(text: str) -> dict[str, str]:
    """Best-effort section splitter for common one-line resume headings."""
    lines = [line.strip() for line in text.splitlines()]
    sections: dict[str, list[str]] = {"other": []}
    current = "other"

    for line in lines:
        key = _heading_to_key(line)
        if key:
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
        if "\n".join(value).strip()
    }


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Create paragraph-aware chunks for lightweight RAG."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text else []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # Split very large paragraphs approximately by sentence.
        pieces = (
            re.split(r"(?<=[.!?])\s+", paragraph)
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            if not piece:
                continue
            if len(current) + len(piece) + 2 <= max_chars:
                current = f"{current}\n{piece}".strip()
            else:
                if current:
                    chunks.append(current)
                current = piece[:max_chars]
                rest = piece[max_chars:]
                while rest:
                    chunks.append(current)
                    current = rest[:max_chars]
                    rest = rest[max_chars:]

    if current:
        chunks.append(current)

    return chunks


def resume_completeness_score(text: str, sections: dict[str, str]) -> tuple[float, list[str]]:
    """A transparent heuristic score; this is not an employer ATS score."""
    checks = {
        "Email/contact info": bool(
            re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
        ),
        "Education section": "education" in sections,
        "Skills section": "skills" in sections,
        "Projects or experience section": (
            "projects" in sections or "experience" in sections
        ),
        "Substantive resume text": len(text.split()) >= 180,
    }
    passed = sum(checks.values())
    score = 100.0 * passed / len(checks)
    missing = [name for name, ok in checks.items() if not ok]
    return score, missing
