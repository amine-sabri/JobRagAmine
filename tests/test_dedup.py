"""Duplicate grouping: the rules, the canonical choice, and the determinism."""

import random

from jobrag.dedup import mark_duplicates


def offer(id_, title_norm, company_norm, description_hash, posted_at=None, city=None, **extra):
    return {
        "id": id_,
        "title": title_norm,
        "title_norm": title_norm,
        "company": company_norm,
        "company_norm": company_norm,
        "description_hash": description_hash,
        "posted_at": posted_at,
        "discovered_at": "2026-01-01T00:00:00+00:00",
        "city": city,
        **extra,
    }


def test_identical_text_same_company_is_a_duplicate():
    offers = [
        offer("b", "dev python", "acme", "h1", "2026-03-01", "Lille"),
        offer("a", "dev python senior", "acme", "h1", "2026-02-01", "Paris"),
    ]
    stats = mark_duplicates(offers)
    canonical = [o for o in offers if o["duplicate_of"] is None]
    assert len(canonical) == 1
    # The oldest row is canonical, whatever the input order.
    assert canonical[0]["id"] == "a"
    assert canonical[0]["duplicate_cities"] == ["Lille", "Paris"]
    assert stats["duplicates"] == 1
    assert stats["rule_hits"] == {"same_text": 1, "same_title": 0}


def test_same_title_and_company_in_two_cities_is_a_duplicate():
    offers = [
        offer("a", "dev web", "leihia", "h1", "2026-02-01", "Paris"),
        offer("b", "dev web", "leihia", "h2", "2026-02-02", "Lyon"),
    ]
    stats = mark_duplicates(offers)
    assert stats["duplicates"] == 1
    assert stats["rule_hits"] == {"same_text": 0, "same_title": 1}
    assert offers[0]["duplicate_of"] is None
    assert offers[1]["duplicate_of"] == "a"
    assert offers[1]["duplicate_cities"] is None


def test_same_title_different_company_is_not_a_duplicate():
    offers = [
        offer("a", "dev web", "acme", "h1", "2026-02-01"),
        offer("b", "dev web", "globex", "h2", "2026-02-02"),
    ]
    stats = mark_duplicates(offers)
    assert stats["duplicates"] == 0
    assert all(o["duplicate_of"] is None for o in offers)
    assert all(o["duplicate_count"] == 0 for o in offers)


def test_the_two_rules_chain_into_one_group():
    # a-b share their text, b-c share their title: one group of three, not two.
    offers = [
        offer("a", "dev python", "acme", "same", "2026-01-01"),
        offer("b", "dev python java", "acme", "same", "2026-01-02"),
        offer("c", "dev python java", "acme", "other", "2026-01-03"),
    ]
    stats = mark_duplicates(offers)
    assert stats["multi_member_groups"] == 1
    assert stats["duplicates"] == 2
    assert {o["duplicate_group"] for o in offers} == {"a"}


def test_canonical_choice_is_deterministic_under_shuffling():
    base = [
        offer(f"id{i}", "dev web", "acme", "h1", f"2026-02-{i + 10}", f"City{i}") for i in range(6)
    ]
    reference = None
    for seed in range(5):
        offers = base.copy()
        random.Random(seed).shuffle(offers)
        offers = [dict(o) for o in offers]
        mark_duplicates(offers)
        canonical = next(o["id"] for o in offers if o["duplicate_of"] is None)
        cities = next(o["duplicate_cities"] for o in offers if o["duplicate_of"] is None)
        assert reference in (None, (canonical, tuple(cities)))
        reference = (canonical, tuple(cities))
    assert reference[0] == "id0"


def test_offers_with_no_posted_at_fall_back_to_discovered_at():
    offers = [
        offer("b", "dev", "acme", "h1", None),
        offer("a", "dev", "acme", "h1", None),
    ]
    offers[0]["discovered_at"] = "2026-01-01T00:00:00+00:00"
    offers[1]["discovered_at"] = "2026-05-01T00:00:00+00:00"
    mark_duplicates(offers)
    assert offers[0]["duplicate_of"] is None  # older discovered_at wins
    assert offers[1]["duplicate_of"] == "b"


def test_a_single_offer_is_its_own_group():
    offers = [offer("a", "dev", "acme", "h1", "2026-01-01")]
    stats = mark_duplicates(offers)
    assert stats == {
        "offers": 1,
        "groups": 1,
        "canonical": 1,
        "duplicates": 0,
        "multi_member_groups": 0,
        "rule_hits": {"same_text": 0, "same_title": 0},
        "largest_groups": [],
    }
    assert offers[0]["duplicate_of"] is None
    assert offers[0]["duplicate_count"] == 0
