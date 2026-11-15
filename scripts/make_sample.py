#!/usr/bin/env python
"""Build data/sample_offers.jsonl: a small, anonymised, committed sample.

The snapshot itself is private and git-ignored. The tests still need realistic
data, so this script picks a spread of offers (every source, both languages,
every contract type, truncated and full text, and one real duplicate group) and
pseudonymises what identifies a company: its name — inside the description too —
its URL path and its id.

Deterministic: same input, same output, so the committed file only changes when
the selection rules change.

Usage:
    python scripts/make_sample.py [--n 50]
"""

import argparse
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.clean import EMAIL_RE, PHONE_RE  # noqa: E402

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "jobrag/sample")
MAX_DESCRIPTION = 1200

# Fields that may contain free text. Timestamps are excluded on purpose: an ISO
# date is not text to anonymise, and scanning it produces false positives.
TEXT_FIELDS = (
    "title",
    "company",
    "description",
    "location",
    "city",
    "region",
    "salary",
    "url",
    "end_client_name",
)

# Hosts that are job boards, so keeping them says nothing about the employer.
# Any other host belongs to the company itself and is replaced.
BOARD_HOSTS = {
    "www.adzuna.fr",
    "candidat.francetravail.fr",
    "job-boards.eu.greenhouse.io",
    "jobs.ashbyhq.com",
    "jobs.smartrecruiters.com",
    "euraxess.ec.europa.eu",
    "emploi.abg.asso.fr",
    "www.welcometothejungle.com",
}


def replace_company(text: str | None, company: str, alias: str) -> str | None:
    """Swap every spelling of `company` for `alias`, on word boundaries."""
    if not text or not company:
        return text
    for name in sorted(
        {company, company.title(), company.upper(), company.lower()}, key=len, reverse=True
    ):
        text = re.sub(rf"\b{re.escape(name)}\b", alias, text, flags=re.IGNORECASE)
    return text


def pick(offers: list[dict], n: int) -> list[dict]:
    """Deterministic, diversity-first selection."""
    offers = sorted(offers, key=lambda o: str(o["id"]))
    chosen: dict[str, dict] = {}

    def take(offer: dict) -> None:
        chosen.setdefault(str(offer["id"]), offer)

    # One real duplicate group, so the dedup tests run on real data.
    biggest = max(offers, key=lambda o: (o["duplicate_count"] or 0, str(o["id"])))
    group = [o for o in offers if o["duplicate_group"] == biggest["duplicate_group"]][:3]
    for offer in group:
        take(offer)

    # At least one of every source, language and contract type.
    for key in ("source_name", "language_code", "contract_label"):
        for value in sorted({o[key] for o in offers}):
            for offer in offers:
                if offer[key] == value and not offer["is_truncated"]:
                    take(offer)
                    break
            else:  # only truncated ones exist for that value
                for offer in offers:
                    if offer[key] == value:
                        take(offer)
                        break

    # Some truncated offers: 63% of the corpus looks like this.
    for offer in offers:
        if len(chosen) >= n - 10:
            break
        if offer["is_truncated"]:
            take(offer)

    # Fill up with full-text offers that have paragraph structure, for chunking.
    for offer in sorted(offers, key=lambda o: (-o["description"].count("\n"), str(o["id"]))):
        if len(chosen) >= n:
            break
        if not offer["is_truncated"]:
            take(offer)

    return sorted(chosen.values(), key=lambda o: str(o["id"]))[:n]


def anonymise(offer: dict, alias: str) -> dict:
    """Replace the company identity and the exact URL; keep everything factual."""
    out = dict(offer)
    company = offer["company"] or ""
    out["id"] = str(uuid.uuid5(NAMESPACE, str(offer["id"])))
    out["external_id"] = None
    out["company"] = alias
    out["company_norm"] = alias.lower()
    out["end_client_name"] = f"{alias} (client)" if offer["end_client_name"] else None

    out["description"] = replace_company(offer["description"][:MAX_DESCRIPTION], company, alias)
    out["description_chars"] = len(out["description"] or "")
    for field in ("title", "location", "region", "salary"):
        out[field] = replace_company(offer[field], company, alias)
    out["title_norm"] = replace_company(offer["title_norm"], company, alias.lower())

    host = (offer["url"] or "").split("//")[-1].split("/")[0]
    if host not in BOARD_HOSTS:
        host = "jobs.example.org"
    out["url"] = f"https://{host}/offers/{out['id'][:8]}"

    for key in ("duplicate_of", "duplicate_group"):
        if out[key]:
            out[key] = str(uuid.uuid5(NAMESPACE, out[key]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="path", default="data/offers.jsonl", type=Path)
    parser.add_argument("--out", default="data/sample_offers.jsonl", type=Path)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{args.path} not found — run `make export` first", file=sys.stderr)
        return 2

    with args.path.open(encoding="utf-8") as fh:
        offers = [json.loads(line) for line in fh]

    selected = pick(offers, args.n)
    aliases = {}
    out: list[dict] = []
    for offer in selected:
        company = offer["company"] or "?"
        alias = aliases.setdefault(company, f"Entreprise {len(aliases) + 1:02d}")
        out.append(anonymise(offer, alias))

    # Refuse to write a sample that still contains a contact detail or a company
    # name. Only free-text fields are scanned; ids and timestamps are not text.
    texts = [str(o[f]) for o in out for f in TEXT_FIELDS if o.get(f)]
    blob = "\n".join(texts)
    leaks = EMAIL_RE.findall(blob) + [m.group(0) for m in PHONE_RE.finditer(blob)]
    real_names = [
        c for c in aliases if len(c) > 3 and re.search(rf"\b{re.escape(c)}\b", blob, re.IGNORECASE)
    ]
    if leaks or real_names:
        print(
            f"refusing to write: leaks={leaks[:3]} company names={real_names[:3]}", file=sys.stderr
        )
        return 1

    args.out.write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8"
    )
    counts = Counter(o["source_name"] for o in out)
    print(f"{len(out)} offers -> {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"sources     {dict(counts)}")
    print(f"languages   {dict(Counter(o['language_code'] for o in out))}")
    print(f"contracts   {dict(Counter(o['contract_label'] for o in out))}")
    print(f"truncated   {sum(o['is_truncated'] for o in out)}/{len(out)}")
    print(
        f"duplicates  {sum(o['duplicate_of'] is not None for o in out)} copies, "
        f"{sum(1 for o in out if o['duplicate_count'])} rows in a group"
    )
    print(f"companies   {len(aliases)} pseudonymised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
