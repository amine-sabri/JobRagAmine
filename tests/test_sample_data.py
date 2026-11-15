"""The committed sample: its privacy contract and its coverage.

This file is what CI runs on, since the real snapshot is git-ignored. If the
sample ever stops being representative, the later phases silently test nothing.
"""

import json
from pathlib import Path

import pytest

from jobrag.clean import EMAIL_RE, PHONE_RE
from jobrag.dedup import mark_duplicates
from jobrag.enums import CONTRACT_LABELS

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_offers.jsonl"
TEXT_FIELDS = ("title", "company", "description", "location", "city", "region", "salary", "url")

REQUIRED_KEYS = {
    "id",
    "source_name",
    "title",
    "title_norm",
    "company",
    "company_norm",
    "city",
    "contract_type",
    "contract_label",
    "language_code",
    "description",
    "description_chars",
    "is_truncated",
    "description_hash",
    "url",
    "posted_at",
    "duplicate_of",
    "duplicate_group",
    "duplicate_count",
}


@pytest.fixture(scope="module")
def offers() -> list[dict]:
    if not SAMPLE.exists():
        pytest.skip("sample not built yet: make sample")
    with SAMPLE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_sample_is_committed_and_small(offers):
    assert len(offers) == 50
    assert SAMPLE.stat().st_size < 512 * 1024  # the pre-commit large-file limit


def test_every_record_has_the_keys_later_phases_rely_on(offers):
    for offer in offers:
        missing = REQUIRED_KEYS - set(offer)
        assert not missing, f"{offer['id']} misses {missing}"


def test_no_contact_detail_survives_in_the_sample(offers):
    for offer in offers:
        for field in TEXT_FIELDS:
            value = offer.get(field) or ""
            assert not EMAIL_RE.search(value), f"e-mail in {offer['id']}.{field}"
            assert not PHONE_RE.search(value), f"phone in {offer['id']}.{field}"


def test_companies_are_pseudonymised(offers):
    for offer in offers:
        assert offer["company"].startswith("Entreprise ")


def test_urls_point_at_a_board_or_a_placeholder(offers):
    for offer in offers:
        assert offer["url"].startswith("https://")
        assert "/offers/" in offer["url"]


def test_every_contract_label_decodes(offers):
    labels = set(CONTRACT_LABELS.values())
    for offer in offers:
        assert offer["contract_label"] in labels


def test_the_sample_covers_what_the_next_phases_need(offers):
    assert {o["language_code"] for o in offers} >= {"fr", "en"}
    assert sum(o["is_truncated"] for o in offers) >= 5
    assert sum(not o["is_truncated"] for o in offers) >= 20
    assert len({o["source_name"] for o in offers}) >= 5
    assert len({o["contract_label"] for o in offers}) >= 4
    # Offers with paragraph structure, for the section-aware chunking of phase 2.
    assert sum(1 for o in offers if o["description"].count("\n\n") >= 2) >= 5


def test_the_sample_still_contains_a_real_duplicate_group(offers):
    stats = mark_duplicates([dict(o) for o in offers])
    assert stats["multi_member_groups"] >= 1
    assert stats["duplicates"] >= 1
