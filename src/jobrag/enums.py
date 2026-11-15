"""Integer enums used by the source platform, decoded to labels.

The source stores small integers. The assistant needs labels: they go into the
prompt, into the citations and into the metadata filters of phase 6, and "CDI"
is what a question says, not "1".

Mappings marked UNVERIFIED are kept as raw integers in the export; nothing
depends on guessing them.
"""

from enum import IntEnum


class ContractType(IntEnum):
    """Values observed in the corpus: 0, 1, 2, 5, 6, 7 (3 and 4 never occur)."""

    UNSPECIFIED = 0
    CDI = 1
    CDD = 2
    INTERNSHIP = 3
    APPRENTICESHIP = 4
    FREELANCE = 5
    INTERIM = 6
    DOCTORATE = 7


CONTRACT_LABELS: dict[int, str] = {
    ContractType.UNSPECIFIED: "non précisé",
    ContractType.CDI: "CDI",
    ContractType.CDD: "CDD",
    ContractType.INTERNSHIP: "stage",
    ContractType.APPRENTICESHIP: "alternance",
    ContractType.FREELANCE: "freelance",
    ContractType.INTERIM: "intérim",
    ContractType.DOCTORATE: "doctorat",
}


class Language(IntEnum):
    """Verified against stopword counts: 0 is French (99.7%), 1 is English (97.8%)."""

    FR = 0
    EN = 1


LANGUAGE_CODES: dict[int, str] = {Language.FR: "fr", Language.EN: "en"}

# Measured from the host of "Url", per Source value (see INTERVIEW_NOTES.md).
SOURCE_NAMES: dict[int, str] = {
    0: "unknown",
    1: "france-travail",
    2: "adzuna",
    10: "greenhouse",
    12: "smartrecruiters",
    13: "ashby",
    14: "recruitee",
    20: "welcome-to-the-jungle",
    50: "abg",
    51: "euraxess",
}

# UNVERIFIED: the platform's own pipeline states. Kept as integers.
STATUS_LABELS: dict[int, str] = {}
LEVEL_LABELS: dict[int, str] = {}


def decode(mapping: dict[int, str], value: int | None, prefix: str) -> str | None:
    """Label for `value`, or a readable fallback like 'status-3' when unmapped."""
    if value is None:
        return None
    label = mapping.get(value)
    return label if label is not None else f"{prefix}-{value}"
