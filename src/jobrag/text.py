"""Text helpers shared by retrieval and by the evaluation tooling.

The stopword list lives here rather than in either caller because both need the
same one: a question is mostly grammar, and "Quelles offres demandent du
Pulumi ?" carries exactly one word worth searching for. Postgres's `french`
configuration does not remove "quelles", "offres" or "demandent" — they are
ordinary words to it — so the query has to be cleaned before it gets there.
"""

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9#+./]+")

#: French and English question scaffolding, plus the job-board vocabulary that
#: appears in nearly every offer and therefore discriminates nothing.
STOPWORDS = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me meme
    mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quelles quels qui sa se
    ses son sur ta te tes toi ton tu un une vos votre vous y est sont ont offre offres poste
    postes emploi emplois recherche recherches recherchent demandent demande demandez combien
    quoi dont ainsi quels quelle cherche cherchent proposent propose utilisent utilise parlent
    parle concernent concerne travaillent travaille font fait
    the a an and are as at be by for from has have in is it its of on or that this to was were
    what which who with job jobs offer offers position positions looking require requires use
    uses using work works about
    """.split()  # noqa: SIM905 - a readable block beats a 140-item list literal
)


def fold(text: str | None) -> str:
    """Lowercase, unaccented form used for every comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenise(text: str | None) -> list[str]:
    return _TOKEN_RE.findall(fold(text))


def content_tokens(text: str | None) -> list[str]:
    """The tokens worth searching on: no stopwords, nothing of length 1.

    >>> content_tokens("Quelles offres demandent du Pulumi ?")
    ['pulumi']
    """
    return [token for token in tokenise(text) if token not in STOPWORDS and len(token) > 1]
