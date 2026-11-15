"""Cleaning applied to every offer before it is indexed.

Order matters: mask contacts first (so no e-mail survives a later transform),
then unwrap markup, then normalise whitespace. Titles get their own
normalisation, used for near-duplicate detection and never shown to the user.
"""

import hashlib
import html
import re
import unicodedata

EMAIL_PLACEHOLDER = "[EMAIL]"
PHONE_PLACEHOLDER = "[PHONE]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# French numbers, written the many ways humans write them:
#   06 12 34 56 78 / 06.12.34.56.78 / 0612345678 / +33 6 12 34 56 78 / +33 (0)6 ...
#
# The separator must be *consistent* across the four groups, or absent. Without
# that, the pattern also matches an ISO timestamp — "02.190606-04" inside
# "2026-10-10T01:16:02.190606-04:00" — and masking would corrupt the text.
PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?:(?:\+33|0033)[\s.\-]?(?:\(0\)[\s.\-]?)?[1-9]|0[1-9])"
    r"(?:\d{8}|(?P<sep>[\s.\-])\d{2}(?:(?P=sep)\d{2}){3})"
    r"(?!\d)"
)

TAG_RE = re.compile(r"<[^>]+>")
BLOCK_TAG_RE = re.compile(r"(?i)<\s*(?:br\s*/?|/\s*(?:p|div|li|ul|ol|h[1-6]|tr))\s*>")

# "(H/F)", "H/F", "F/H", "(h/f/d)", "M/F" — and the "(H/F)" glued to a word.
GENDER_MARKER_RE = re.compile(
    r"[\(\[]?\b([hfmw])\s*[/.\-]\s*([hfmwd])(?:\s*[/.\-]\s*([hfmwd]))?\b[\)\]]?",
    re.IGNORECASE,
)
NON_WORD_RE = re.compile(r"[^0-9a-z#+]+")
WS_RE = re.compile(r"[ \t   ]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def mask_contacts(text: str) -> tuple[str, int, int]:
    """Replace e-mail addresses and phone numbers. Returns (text, n_emails, n_phones).

    Counts are reported by the export so the masking can be audited instead of
    trusted.
    """
    if not text:
        return "", 0, 0
    text, n_emails = EMAIL_RE.subn(EMAIL_PLACEHOLDER, text)
    text, n_phones = PHONE_RE.subn(PHONE_PLACEHOLDER, text)
    return text, n_emails, n_phones


def html_to_text(text: str) -> str:
    """Unwrap the small share of offers (~0.3%) that arrive with HTML markup."""
    if not text:
        return ""
    if "<" not in text and "&" not in text:
        return text
    text = BLOCK_TAG_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines, keeping paragraph structure.

    Paragraph breaks survive on purpose: phase 2 uses them to find sections.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def clean_description(raw: str) -> tuple[str, int, int]:
    """Full description pipeline. Returns (text, n_emails_masked, n_phones_masked)."""
    masked, n_emails, n_phones = mask_contacts(raw)
    return normalise_whitespace(html_to_text(masked)), n_emails, n_phones


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_title(title: str) -> str:
    """Comparison key for a job title: lowercase, unaccented, no gender marker.

    >>> normalise_title("Développeur Web Fullstack (H/F) · Paris")
    'developpeur web fullstack'
    """
    if not title:
        return ""
    text = strip_accents(title).lower()
    # Drop a trailing " · Paris" / " - Lille" style location or contract suffix.
    text = text.split("·")[0]
    text = GENDER_MARKER_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


LEGAL_FORMS = frozenset({"sas", "sa", "sarl", "sasu", "eurl", "scop", "inc", "ltd", "gmbh"})


def _merge_initials(words: list[str]) -> list[str]:
    """Glue runs of single letters back together: [s, a, r, l] -> [sarl].

    Dotted acronyms ("ACME S.A.R.L.", "H&M") survive punctuation stripping as
    isolated letters; without this, a legal form can no longer be recognised.
    """
    merged: list[str] = []
    run: list[str] = []
    for word in words:
        if len(word) == 1 and word.isalpha():
            run.append(word)
            continue
        if run:
            merged.append("".join(run))
            run = []
        merged.append(word)
    if run:
        merged.append("".join(run))
    return merged


def normalise_company(name: str) -> str:
    """Comparison key for a company name, ignoring legal forms and punctuation."""
    if not name:
        return ""
    text = strip_accents(name).lower()
    text = NON_WORD_RE.sub(" ", text)
    words = _merge_initials(text.split())
    return " ".join(w for w in words if w not in LEGAL_FORMS)


def content_hash(*parts: str) -> str:
    """Stable hash of normalised content, used to group exact duplicates."""
    joined = "\u001f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]
