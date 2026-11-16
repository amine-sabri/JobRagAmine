"""Turning a question into metadata filters, carefully.

"Quelles offres en CDI à Lille ?" should restrict the search to `city = 'Lille'
AND contract_type = 1` — a filter the vectors cannot express, since a chunk
about Lille and a chunk about Lyon look almost identical to an embedding.

**The risk is silent over-filtering.** A wrong filter does not produce a wrong
answer, it produces *no* answer, and the assistant then says "je ne sais pas"
about something the corpus contains. Four guards follow from that:

1. **A city needs a locative preposition** ("à Lille", "dans le Nord", "près de
   Lyon"). Without one, "Quelles offres chez Nantes Métropole Habitat ?" would
   filter on `city = 'Nantes'` and drop the employer's real postings.
2. **Two cities means no city filter.** "à Lille ou à Paris" is a question the
   filter cannot express, and picking one silently answers a different
   question.
3. **A negation disables its filter.** "sans télétravail" must not become
   `is_remote = true`; it becomes no filter, with a note.
4. **Every filter is reported**, and the caller may relax them: if a filtered
   search finds nothing, `/ask` retries without the filters and says so, rather
   than refusing.
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from .text import fold

#: Contract vocabulary -> ContractType. Order matters: the longest wins.
CONTRACT_PATTERNS: list[tuple[str, int]] = [
    (r"\bcdi\b|\bcontrats? (?:a|à) dur(?:e|é)e ind(?:e|é)termin(?:e|é)e\b", 1),
    (r"\bcdd\b|\bcontrats? (?:a|à) dur(?:e|é)e d(?:e|é)termin(?:e|é)e\b", 2),
    (r"\bstages?\b|\bstagiaires?\b|\binternships?\b", 3),
    (r"\balternances?\b|\bapprentissages?\b|\balternants?\b", 4),
    (r"\bfreelances?\b|\bind(?:e|é)pendants?\b|\bprestations?\b", 5),
    (r"\bint(?:e|é)rims?\b|\bmissions? d.int(?:e|é)rim\b", 6),
    (r"\bdoctorats?\b|\bth(?:e|è)ses?\b|\bphd\b|\bdoctoral\b", 7),
]

REMOTE_PATTERN = r"\bt(?:e|é)l(?:e|é)travail\b|\bremote\b|\b(?:a|à) distance\b|\bhome ?office\b"
NEGATION_PATTERN = r"\b(?:sans|pas de|aucun|non|no|without)\b[^.?!]{0,20}$"

#: Prepositions that make the following word a place rather than a name.
LOCATIVE = (
    r"(?:(?:a|à)|dans|sur|pr(?:e|è)s de|autour de|proche de|vers|in|near|around)"
    r"\s+(?:la |le |les |l'|the )?"
)

RECENCY_PATTERNS: list[tuple[str, int]] = [
    (r"\baujourd.hui\b|\bce jour\b|\btoday\b", 1),
    (r"\bcette semaine\b|\bdepuis une semaine\b|\b7 jours\b|\bthis week\b", 7),
    (r"\bdepuis (\d+) jours?\b|\bces (\d+) jours?\b|\blast (\d+) days?\b", 0),  # n captured
    (r"\bce mois\b|\bdepuis un mois\b|\b30 jours\b|\bthis month\b", 30),
    (r"\br(?:e|é)centes?\b|\br(?:e|é)cents?\b|\bderni(?:e|è)res?\b|\brecent\b", 30),
]


@dataclass(frozen=True)
class Filters:
    """Metadata restrictions extracted from a question."""

    city: str | None = None
    contract_type: int | None = None
    is_remote: bool | None = None
    posted_after: date | None = None
    language: str | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, name) is None
            for name in ("city", "contract_type", "is_remote", "posted_after", "language")
        )

    def as_conditions(self) -> tuple[list[str], dict[str, Any]]:
        """SQL conditions on the `offers` alias `o`, plus their parameters."""
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if self.city:
            conditions.append("lower(o.city) = lower(%(f_city)s)")
            params["f_city"] = self.city
        if self.contract_type is not None:
            conditions.append("o.contract_type = %(f_contract)s")
            params["f_contract"] = self.contract_type
        if self.is_remote is not None:
            conditions.append("o.is_remote = %(f_remote)s")
            params["f_remote"] = self.is_remote
        if self.posted_after is not None:
            conditions.append("o.posted_at >= %(f_posted_after)s")
            params["f_posted_after"] = self.posted_after
        if self.language:
            conditions.append("o.language_code = %(f_language)s")
            params["f_language"] = self.language
        return conditions, params

    def describe(self) -> str:
        parts = []
        if self.city:
            parts.append(f"ville={self.city}")
        if self.contract_type is not None:
            parts.append(f"contrat={self.contract_type}")
        if self.is_remote is not None:
            parts.append(f"télétravail={'oui' if self.is_remote else 'non'}")
        if self.posted_after:
            parts.append(f"publiée depuis {self.posted_after.isoformat()}")
        if self.language:
            parts.append(f"langue={self.language}")
        return ", ".join(parts) or "aucun"


@dataclass
class Extraction:
    filters: Filters
    notes: list[str] = field(default_factory=list)


def _negated(folded: str, match_start: int) -> bool:
    """Is the match preceded by a negation, within the same clause?"""
    before = folded[:match_start]
    return bool(re.search(NEGATION_PATTERN, before))


def extract_filters(
    question: str,
    *,
    cities: set[str] | None = None,
    today: date | None = None,
) -> Extraction:
    """Extract city / contract / remote / recency / language from a question."""
    folded = fold(question)
    notes: list[str] = []
    today = today or date.today()

    contract_type = None
    for pattern, value in CONTRACT_PATTERNS:
        match = re.search(pattern, folded)
        if match:
            if _negated(folded, match.start()):
                notes.append(f"contract {value} mentioned but negated: no contract filter")
                break
            contract_type = value
            break

    is_remote = None
    remote_match = re.search(REMOTE_PATTERN, folded)
    if remote_match:
        if _negated(folded, remote_match.start()):
            notes.append("télétravail is negated: no remote filter (the schema has no 'on site')")
        else:
            is_remote = True

    city = None
    if cities:
        folded_cities = {fold(name): name for name in cities if name}
        found: list[str] = []
        for match in re.finditer(rf"{LOCATIVE}([a-z][a-z\-' ]{{1,28}}?)(?=\s|[,.?!]|$)", folded):
            candidate = match.group(1).strip()
            # Longest match first: "saint etienne" before "saint".
            for length in range(len(candidate), 1, -1):
                prefix = candidate[:length].strip()
                if prefix in folded_cities:
                    found.append(folded_cities[prefix])
                    break
        unique = list(dict.fromkeys(found))
        if len(unique) == 1:
            city = unique[0]
        elif len(unique) > 1:
            notes.append(f"several cities mentioned ({', '.join(unique)}): no city filter")

    posted_after = None
    for pattern, days in RECENCY_PATTERNS:
        match = re.search(pattern, folded)
        if match:
            captured = next((g for g in match.groups() if g), None)
            window = int(captured) if captured else days
            if window:
                posted_after = today - timedelta(days=window)
            break

    language = None
    if re.search(r"\ben anglais\b|\bin english\b", folded):
        language = "en"
    elif re.search(r"\ben fran(?:c|ç)ais\b|\bin french\b", folded):
        language = "fr"

    return Extraction(
        filters=Filters(
            city=city,
            contract_type=contract_type,
            is_remote=is_remote,
            posted_after=posted_after,
            language=language,
        ),
        notes=notes,
    )


@lru_cache(maxsize=1)
def known_cities() -> frozenset[str]:
    """Every city present in the index, for matching a question against reality."""
    from .db import connection

    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT city FROM offers WHERE city IS NOT NULL AND city <> ''")
        return frozenset(row[0] for row in cur.fetchall())
