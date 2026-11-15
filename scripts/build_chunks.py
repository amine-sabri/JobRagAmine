#!/usr/bin/env python
"""Chunk data/offers.jsonl into data/chunks.jsonl.

Only canonical offers are chunked by default: a position posted in 82 cities
must not occupy 82 slots in the top-k. The copies stay in the export, and the
canonical row carries their cities.

Writes eval/results/chunking_stats.{md,json} and injects the table into
README.md between the `chunking-stats` markers.

Usage:
    python scripts/build_chunks.py [--max-chars 1200] [--overlap 150]
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.chunking import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP,
    chunk_offer,
)
from jobrag.report import inject_into_file, markdown_table  # noqa: E402


def percentiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="path", default="data/offers.jsonl", type=Path)
    parser.add_argument("--out", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="also chunk the flagged copies (default: canonical offers only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the statistics without writing chunks, results or README",
    )
    parser.add_argument("--results", default="eval/results", type=Path)
    parser.add_argument("--readme", default="README.md", type=Path)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{args.path} not found — run `make export` first", file=sys.stderr)
        return 2

    started = time.perf_counter()
    offers = skipped = 0
    per_offer: list[int] = []
    lengths: list[int] = []
    sections: Counter = Counter()
    truncated_chunks = 0
    truncated_offers = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + (".dry" if args.dry_run else ".tmp"))
    with args.path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            offer = json.loads(line)
            if offer["duplicate_of"] is not None and not args.include_duplicates:
                skipped += 1
                continue
            chunks = chunk_offer(
                offer,
                max_chars=args.max_chars,
                overlap=args.overlap,
                min_chars=args.min_chars,
            )
            offers += 1
            truncated_offers += bool(offer["is_truncated"])
            per_offer.append(len(chunks))
            for chunk in chunks:
                lengths.append(chunk["n_chars"])
                sections[chunk["section"] or "(aucune)"] += 1
                truncated_chunks += chunk["is_truncated"]
                dst.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    if args.dry_run:
        tmp.unlink()
    else:
        tmp.replace(args.out)
    elapsed = time.perf_counter() - started

    if max(lengths) > args.max_chars:
        print(f"BUG: a chunk of {max(lengths)} chars exceeds the budget", file=sys.stderr)
        return 1

    with_section = sum(count for label, count in sections.items() if label != "(aucune)")
    stats = {
        "command": (
            f"python scripts/build_chunks.py --max-chars {args.max_chars} --overlap {args.overlap}"
        ),
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "min_chars": args.min_chars,
        "offers_chunked": offers,
        "duplicate_offers_skipped": skipped,
        "truncated_offers": truncated_offers,
        "chunks": len(lengths),
        "chunks_per_offer": percentiles(per_offer),
        "chunk_chars": percentiles(lengths),
        "chunks_with_a_section": with_section,
        "sections": dict(sections.most_common()),
        "truncated_chunks": truncated_chunks,
        "seconds": round(elapsed, 1),
    }

    size_table = markdown_table(
        ["", "min", "p50", "p75", "p95", "max", "mean"],
        [
            [
                "chunks per offer",
                *[
                    stats["chunks_per_offer"][k]
                    for k in ("min", "p50", "p75", "p95", "max", "mean")
                ],
            ],
            [
                "characters per chunk",
                *[stats["chunk_chars"][k] for k in ("min", "p50", "p75", "p95", "max", "mean")],
            ],
        ],
    )
    section_table = markdown_table(
        ["section", "chunks", "share"],
        [[label, count, f"{count / len(lengths):.0%}"] for label, count in sections.most_common()],
    )

    md = f"""<!-- generated by: `{stats["command"]}` -->
**{offers} canonical offers → {len(lengths)} chunks** in {elapsed:.1f}s
(`max_chars={args.max_chars}`, `overlap={args.overlap}`, `min_chars={args.min_chars}`).
{skipped} flagged copies were not chunked. {truncated_chunks} chunks
({truncated_chunks / len(lengths):.0%}) come from an offer truncated at the source and are flagged
as such. No chunk exceeds the budget: the longest is {max(lengths)} characters, header and overlap
included.

{size_table}

{section_table}
"""
    if args.dry_run:
        print(md)
        print(f"dry run: {len(lengths)} chunks, nothing written")
        return 0

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "chunking_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    (args.results / "chunking_stats.md").write_text(md, encoding="utf-8")
    inject_into_file(args.readme, "chunking-stats", md)
    print(md)
    print(
        f"written: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), {args.results}/, README.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
