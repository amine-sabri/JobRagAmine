"""Splitting an offer's description into indexable chunks.

Three rules decide the design:

1. **Nothing may exceed the embedding model's input.** e5-base stops at 512
   tokens and silently drops the rest, so the character budget is a hard
   guarantee: `len(chunk["text"]) <= max_chars`, header and overlap included.
2. **A chunk must stand alone.** It is what retrieval returns and what the LLM
   reads, so every chunk starts with a header (`title · company · city ·
   contract`). Without it, "3 ans d'expérience exigés" cannot be cited.
3. **Sections first, size second.** Only 47% of the full-text offers have a
   recognisable heading (measured), so section splitting is an optimisation with
   a paragraph/sentence fallback, not the main mechanism.
"""

import re
from typing import Any

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150
DEFAULT_MIN_CHARS = 200

# Canonical section labels, with the French and English headings that map to
# them. Matched on a short line of its own, with or without a colon.
# Canonical section labels, with the French and English headings that map to
# them. Matched on a short line of its own, with or without a colon.
SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    "missions": (
        r"(?:vos? |les |tes )?missions?",
        r"t[âa]ches",
        r"le poste",
        r"votre r[ôo]le",
        r"responsibilities",
        r"what you(?:'ll| will) do",
        r"the role",
        r"job description",
    ),
    "profil": (
        r"(?:votre |le |ton )?profil(?: recherch[ée])?",
        r"qui (?:[êe]tes|es)[- ](?:tu|vous)",
        r"candidat id[ée]al",
        r"about you",
        r"who you are",
        r"your profile",
    ),
    "competences": (
        r"comp[ée]tences?(?: requises| attendues| techniques)?",
        r"savoir[- ]faire",
        r"pr[ée]requis",
        r"exigences",
        r"qualifications",
        r"requirements",
        r"skills",
        r"must[- ]have",
        r"nice[- ]to[- ]have",
        r"stack technique",
    ),
    "experience": (
        r"exp[ée]riences?(?: requise| souhait[ée]e)?",
        r"formation(?: et exp[ée]rience)?",
        r"dipl[ôo]mes?",
        r"education",
    ),
    "avantages": (
        r"avantages?",
        r"ce que nous (?:offrons|proposons)",
        r"nous vous (?:offrons|proposons)",
        r"r[ée]mun[ée]ration",
        r"salaire",
        r"package",
        r"benefits",
        r"what we offer",
        r"perks",
    ),
    "entreprise": (
        r"qui sommes[- ]nous",
        r"(?:l'|notre )entreprise",
        r"[àa] propos",
        r"pr[ée]sentation de l'entreprise",
        r"about (?:us|the company)",
        r"our company",
    ),
    "processus": (
        r"processus(?: de recrutement)?",
        r"d[ée]roulement",
        r"[ée]tapes? du recrutement",
        r"comment postuler",
        r"hiring process",
        r"how to apply",
        r"recruitment process",
    ),
}

_ALL_PATTERNS = "|".join(p for patterns in SECTION_HEADINGS.values() for p in patterns)

_HEADING_RE = re.compile(
    r"^[\s\-–•*#>]*(?:\d+[.)]\s*)?(?:" + _ALL_PATTERNS + r")\s*:?\s*$",
    re.IGNORECASE,
)
_LABEL_RES = [
    (label, re.compile("|".join(patterns), re.IGNORECASE))
    for label, patterns in SECTION_HEADINGS.items()
]

# A heading is a *short* line: "Missions :" is one, a sentence starting with
# "Vos missions seront variées et …" is not.
MAX_HEADING_CHARS = 80

_SENTENCE_RE = re.compile(r"(?<=[.!?…:])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def label_for(line: str) -> str | None:
    """The canonical section a heading line belongs to, or None."""
    if len(line) > MAX_HEADING_CHARS or not _HEADING_RE.match(line.strip()):
        return None
    stripped = line.strip().strip(":").strip()
    for label, pattern in _LABEL_RES:
        if pattern.fullmatch(re.sub(r"^[\s\-–•*#>]*(?:\d+[.)]\s*)?", "", stripped)):
            return label
    return None


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split a description on its headings.

    Text before the first heading keeps the label None (the intro). Headings are
    dropped from the body: the label carries the information, and repeating
    "Missions :" inside the chunk only wastes budget.
    """
    if not text.strip():
        return []
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.split("\n"):
        label = label_for(line)
        if label is not None:
            sections.append((label, []))
        else:
            sections[-1][1].append(line)
    out = []
    for label, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            out.append((label, body))
    return out


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def _hard_split(text: str, budget: int) -> list[str]:
    """Cut `text` into pieces of at most `budget` characters, on a word boundary."""
    pieces = []
    rest = text
    while len(rest) > budget:
        cut = rest.rfind(" ", 0, budget)
        if cut <= budget // 2:  # one very long token: cut it mid-word
            cut = budget
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _atoms(text: str, budget: int) -> list[str]:
    """Indivisible pieces: paragraphs, then sentences, then a hard cut."""
    out: list[str] = []
    for paragraph in _paragraphs(text):
        if len(paragraph) <= budget:
            out.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE_RE.split(paragraph):
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= budget:
                current = candidate
                continue
            if current:
                out.append(current)
            current = ""
            if len(sentence) <= budget:
                current = sentence
            else:
                out.extend(_hard_split(sentence, budget))
        if current:
            out.append(current)
    return out


def build_header(offer: dict[str, Any]) -> str:
    """The line prefixed to every chunk of an offer, so it can stand alone."""
    parts = [offer.get("title") or "poste non précisé"]
    if offer.get("company"):
        parts.append(offer["company"])
    if offer.get("city"):
        parts.append(offer["city"])
    elif offer.get("location"):
        parts.append(offer["location"])
    contract = offer.get("contract_label")
    if contract and contract != "non précisé":
        parts.append(contract)
    if offer.get("is_remote"):
        parts.append("télétravail")
    return " · ".join(str(p) for p in parts)


def _tail(text: str, budget: int) -> str:
    """The last `budget` characters of `text`, starting at a word boundary."""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    tail = text[-budget:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_offer(
    offer: dict[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[dict[str, Any]]:
    """Split one exported offer into chunks. Never returns an empty list.

    An offer with no description still produces one header-only chunk: it stays
    findable by title, company and city, which is what a truncated or empty
    offer can honestly answer.
    """
    header = build_header(offer)
    header_block = f"{header}\n\n"
    # The smallest budget any window can have: every atom must fit in it.
    atom_budget = max(max_chars - len(header_block) - overlap, 80)

    atoms: list[tuple[str | None, str]] = []
    for section, body in split_sections(offer.get("description") or ""):
        atoms.extend((section, atom) for atom in _atoms(body, atom_budget))

    windows: list[list[tuple[str | None, str]]] = []
    current: list[tuple[str | None, str]] = []
    current_len = 0
    for section, text in atoms:
        budget = max_chars - len(header_block) - (overlap if windows or current_len else 0)
        if current:
            addition = len(text) + 2  # the "\n\n" joining atoms
            starts_new_section = current[-1][0] != section
            if current_len + addition > budget or (starts_new_section and current_len >= min_chars):
                windows.append(current)
                current, current_len = [], 0
        current.append((section, text))
        current_len += len(text) + (2 if len(current) > 1 else 0)
    if current:
        windows.append(current)

    # A tiny last window reads badly and retrieves badly: grow it by taking
    # atoms back from the window before it, while they fit.
    while len(windows) > 1:
        last_len = sum(len(t) + 2 for _, t in windows[-1]) - 2
        if last_len >= min_chars or len(windows[-2]) < 2:
            break
        moved = windows[-2].pop()
        if last_len + len(moved[1]) + 2 > max_chars - len(header_block) - overlap:
            windows[-2].append(moved)
            break
        windows[-1].insert(0, moved)

    if not windows:
        windows = [[]]

    chunks: list[dict[str, Any]] = []
    previous_body = ""
    for index, window in enumerate(windows):
        body = "\n\n".join(text for _, text in window)
        prefix = ""
        if index and overlap > 2:
            tail = _tail(previous_body, overlap - 2)
            prefix = f"{tail}\n\n" if tail else ""
        text = f"{header_block}{prefix}{body}".strip()
        sections = list(dict.fromkeys(section for section, _ in window if section))
        chunks.append(
            {
                "chunk_id": f"{offer['id']}:{index}",
                "offer_id": offer["id"],
                "chunk_index": index,
                "section": sections[0] if sections else None,
                "sections": sections,
                "text": text,
                "n_chars": len(text),
                "header_chars": len(header_block),
                "is_truncated": bool(offer.get("is_truncated")),
                "duplicate_group": offer.get("duplicate_group"),
            }
        )
        previous_body = body
    return chunks
