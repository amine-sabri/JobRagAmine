"""Near-duplicate detection.

Job offers are posted once per city and re-posted on several boards, so the same
position appears many times. Two rules, deliberately simple and auditable:

  A. same cleaned description + same company  -> the same text, posted twice
  B. same normalised title + same company     -> the same position, one row per city

A union-find merges the two rules, so a chain (A links x-y, B links y-z) ends up
in one group. Copies are kept, flagged with `duplicate_of`: throwing them away
would lose the cities a position is open in, which questions do ask about.

The canonical row of a group is the oldest one (posted_at, else discovered_at),
with the id as tie-break, so the choice is deterministic across runs.
"""

from collections import defaultdict
from typing import Any

Offer = dict[str, Any]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:  # path compression
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smaller id becomes the root: deterministic, independent of input order.
            if rb < ra:
                ra, rb = rb, ra
            self._parent[rb] = ra


def _sort_key(offer: Offer) -> tuple[str, str]:
    date = offer.get("posted_at") or offer.get("discovered_at") or ""
    return (str(date), str(offer["id"]))


def mark_duplicates(offers: list[Offer]) -> dict[str, Any]:
    """Annotate `offers` in place and return statistics.

    Adds to every offer: `duplicate_of` (None on the canonical row),
    `duplicate_group` (the canonical id), `duplicate_count` (group size - 1).
    The canonical row also gets `duplicate_cities`: every city of the group.
    """
    uf = _UnionFind()
    by_text: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_title: dict[tuple[str, str], list[str]] = defaultdict(list)
    index: dict[str, Offer] = {}

    for offer in offers:
        key = str(offer["id"])
        index[key] = offer
        uf.add(key)
        company = offer.get("company_norm") or ""
        by_text[(offer["description_hash"], company)].append(key)
        by_title[(offer.get("title_norm") or "", company)].append(key)

    rule_hits = {"same_text": 0, "same_title": 0}
    for name, buckets in (("same_text", by_text), ("same_title", by_title)):
        for keys in buckets.values():
            if len(keys) > 1:
                rule_hits[name] += len(keys) - 1
                first = keys[0]
                for other in keys[1:]:
                    uf.union(first, other)

    groups: dict[str, list[Offer]] = defaultdict(list)
    for key, offer in index.items():
        groups[uf.find(key)].append(offer)

    n_duplicates = 0
    sizes: list[tuple[int, str, str]] = []
    for members in groups.values():
        members.sort(key=_sort_key)
        canonical = members[0]
        canonical_id = str(canonical["id"])
        cities = sorted({m.get("city") for m in members if m.get("city")})
        for member in members:
            member["duplicate_group"] = canonical_id
            member["duplicate_count"] = len(members) - 1
            member["duplicate_of"] = None if member is canonical else canonical_id
            member["duplicate_cities"] = cities if member is canonical else None
        if len(members) > 1:
            n_duplicates += len(members) - 1
            sizes.append((len(members), canonical.get("title", ""), canonical.get("company", "")))

    sizes.sort(reverse=True)
    return {
        "offers": len(offers),
        "groups": len(groups),
        "canonical": len(groups),
        "duplicates": n_duplicates,
        "multi_member_groups": len(sizes),
        "rule_hits": rule_hits,
        "largest_groups": [
            {"size": size, "title": title, "company": company}
            for size, title, company in sizes[:10]
        ],
    }
