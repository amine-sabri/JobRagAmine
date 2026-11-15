#!/usr/bin/env python
"""Describe data/offers.jsonl: coverage, duplicates, truncation, lengths.

Writes eval/results/corpus_stats.{md,json} and injects the summary tables into
README.md between the `corpus-stats` markers.

Usage:
    python scripts/corpus_stats.py [--in data/offers.jsonl]
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.report import inject_into_file, markdown_table  # noqa: E402

COMMAND = "python scripts/corpus_stats.py"


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="path", default="data/offers.jsonl", type=Path)
    parser.add_argument("--results", default="eval/results", type=Path)
    parser.add_argument("--readme", default="README.md", type=Path)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{args.path} not found — run `make export` first", file=sys.stderr)
        return 2

    total = 0
    canonical = 0
    truncated = 0
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_contract: Counter = Counter()
    by_language: Counter = Counter()
    by_status: Counter = Counter()
    by_level: Counter = Counter()
    coverage: Counter = Counter()
    lengths_full: list[int] = []
    lengths_truncated: list[int] = []
    masked = Counter()
    cities: set[str] = set()
    companies: set[str] = set()
    posted: list[str] = []
    groups: list[tuple[int, str, str, int]] = []

    with args.path.open(encoding="utf-8") as fh:
        for line in fh:
            offer = json.loads(line)
            total += 1
            is_canonical = offer["duplicate_of"] is None
            canonical += is_canonical
            truncated += offer["is_truncated"]
            source = offer["source_name"]
            by_source[source]["offers"] += 1
            by_source[source]["canonical"] += is_canonical
            by_source[source]["truncated"] += offer["is_truncated"]
            by_source[source]["chars"] += offer["description_chars"]
            masked["emails"] += offer["masked_emails"]
            masked["phones"] += offer["masked_phones"]
            if is_canonical:
                by_contract[offer["contract_label"]] += 1
                by_language[offer["language_code"]] += 1
                by_status[offer["status"]] += 1
                by_level[offer["level"]] += 1
                coverage["city"] += bool(offer["city"])
                coverage["salary"] += bool(offer["salary"])
                coverage["remote"] += bool(offer["is_remote"])
                coverage["intermediary"] += bool(offer["posted_by_intermediary"])
                coverage["posted_at"] += bool(offer["posted_at"])
                if offer["city"]:
                    cities.add(offer["city"])
                if offer["company"]:
                    companies.add(offer["company"])
                if offer["posted_at"]:
                    posted.append(offer["posted_at"])
                (lengths_truncated if offer["is_truncated"] else lengths_full).append(
                    offer["description_chars"]
                )
                if offer["duplicate_count"]:
                    groups.append(
                        (
                            offer["duplicate_count"] + 1,
                            offer["title"],
                            offer["company"],
                            len(offer["duplicate_cities"] or []),
                        )
                    )

    groups.sort(reverse=True)
    indexable = len(lengths_full)
    stats = {
        "command": COMMAND,
        "total_rows": total,
        "canonical": canonical,
        "duplicate_copies": total - canonical,
        "truncated_rows": truncated,
        "indexable_full_text": indexable,
        "distinct_companies": len(companies),
        "distinct_cities": len(cities),
        "posted_at_range": [min(posted)[:10], max(posted)[:10]] if posted else None,
        "masked": dict(masked),
        "length_full_text": percentiles(lengths_full),
        "length_truncated": percentiles(lengths_truncated),
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "by_contract": dict(by_contract),
        "by_language": dict(by_language),
        "by_status_raw": dict(by_status),
        "by_level_raw": dict(by_level),
        "coverage": dict(coverage),
        "largest_duplicate_groups": [
            {"size": s, "title": t, "company": c, "cities": n} for s, t, c, n in groups[:10]
        ],
    }

    source_table = markdown_table(
        ["source", "offers", "canonical", "truncated", "truncated %", "mean chars"],
        [
            [
                name,
                c["offers"],
                c["canonical"],
                c["truncated"],
                f"{c['truncated'] / c['offers']:.0%}",
                round(c["chars"] / c["offers"]),
            ]
            for name, c in sorted(by_source.items(), key=lambda kv: -kv[1]["offers"])
        ],
    )
    contract_table = markdown_table(
        ["contract", "canonical offers"],
        [[k, v] for k, v in sorted(by_contract.items(), key=lambda kv: -kv[1])],
    )
    length_table = markdown_table(
        ["description chars", "min", "p25", "p50", "p75", "p95", "max", "mean"],
        [
            [
                "full text",
                *[
                    stats["length_full_text"].get(k)
                    for k in ("min", "p25", "p50", "p75", "p95", "max", "mean")
                ],
            ],
            [
                "truncated at source",
                *[
                    stats["length_truncated"].get(k)
                    for k in ("min", "p25", "p50", "p75", "p95", "max", "mean")
                ],
            ],
        ],
    )
    duplicate_table = markdown_table(
        ["copies", "cities", "title", "company"],
        [[s, n, t, c] for s, t, c, n in groups[:5]],
    )

    md = f"""<!-- generated by: `{COMMAND}` -->
**{total} rows** exported, **{canonical}** canonical after deduplication
({total - canonical} copies flagged with `duplicate_of`), **{truncated}** truncated by the source
board ({truncated / total:.0%}). Usable full-text corpus: **{indexable} offers**.
{len(companies)} companies, {len(cities)} cities, posted between
{stats["posted_at_range"][0]} and {stats["posted_at_range"][1]}.
Masked before indexing: {masked["emails"]} e-mail addresses, {masked["phones"]} phone numbers.

{source_table}

{length_table}

{contract_table}

Largest duplicate groups (same position, one row per city):

{duplicate_table}
"""

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    (args.results / "corpus_stats.md").write_text(md, encoding="utf-8")
    inject_into_file(args.readme, "corpus-stats", md)
    print(md)
    print(f"written: {args.results}/corpus_stats.md, corpus_stats.json; README.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
