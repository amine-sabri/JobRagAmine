"""Chunking: the hard guarantees, the section logic, and the fallbacks."""

import json
from pathlib import Path

import pytest

from jobrag.chunking import (
    DEFAULT_MAX_CHARS,
    build_header,
    chunk_offer,
    label_for,
    split_sections,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_offers.jsonl"


def make_offer(description: str, **extra) -> dict:
    return {
        "id": "3f1b2c4d-0000-4000-8000-000000000001",
        "title": "Développeur Python (H/F)",
        "company": "ACME",
        "city": "Lille",
        "contract_label": "CDI",
        "is_remote": False,
        "is_truncated": False,
        "duplicate_group": "3f1b2c4d-0000-4000-8000-000000000001",
        "description": description,
        **extra,
    }


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Missions :", "missions"),
        ("MISSIONS", "missions"),
        ("- Vos missions :", "missions"),
        ("1. Le poste", "missions"),
        ("Votre profil", "profil"),
        ("Profil recherché :", "profil"),
        ("Compétences requises", "competences"),
        ("Stack technique", "competences"),
        ("Avantages", "avantages"),
        ("Qui sommes-nous ?", None),  # the trailing "?" is not a heading form
        ("What you will do", "missions"),
        ("Requirements", "competences"),
        # Not headings: sentences that merely start with the same word.
        ("Vos missions seront variées et nombreuses sur ce poste.", None),
        ("Nous recherchons un profil autonome capable de travailler seul.", None),
        ("", None),
    ],
)
def test_heading_detection(line, expected):
    assert label_for(line) == expected


def test_sections_are_split_and_headings_dropped():
    description = (
        "Nous recrutons.\n\nMissions :\nDévelopper des API.\n\n"
        "Profil recherché :\nBac+5 en informatique.\n\nAvantages\nTickets restaurant."
    )
    sections = split_sections(description)
    assert [label for label, _ in sections] == [None, "missions", "profil", "avantages"]
    assert sections[1][1] == "Développer des API."
    # The heading line itself must not stay in the body: it would waste budget.
    assert "Missions :" not in "".join(body for _, body in sections)


def test_intro_before_the_first_heading_is_kept():
    sections = split_sections("Présentation de 3 lignes.\n\nMissions :\nCoder.")
    assert sections[0] == (None, "Présentation de 3 lignes.")


def test_no_chunk_exceeds_the_budget_header_and_overlap_included():
    # A long description with no paragraph break and very long sentences: the
    # worst case for the splitter.
    description = ("Une phrase interminable sans aucune ponctuation " * 60).strip()
    chunks = chunk_offer(make_offer(description))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["n_chars"] <= DEFAULT_MAX_CHARS, chunk["n_chars"]


def test_every_chunk_starts_with_the_header_so_it_stands_alone():
    chunks = chunk_offer(make_offer("Missions :\n" + "Développer. " * 300))
    header = build_header(make_offer(""))
    for chunk in chunks:
        assert chunk["text"].startswith(header)
        assert chunk["header_chars"] == len(header) + 2


def test_no_text_is_silently_lost():
    description = "Missions :\n" + " ".join(f"point{i}." for i in range(400))
    chunks = chunk_offer(make_offer(description))
    joined = " ".join(chunk["text"] for chunk in chunks)
    for i in range(400):
        assert f"point{i}." in joined, f"point{i} disappeared"


def test_consecutive_chunks_overlap():
    description = "Missions :\n" + " ".join(f"mot{i}" for i in range(500))
    chunks = chunk_offer(make_offer(description), overlap=150)
    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        body_previous = previous["text"][previous["header_chars"] :]
        body_current = current["text"][current["header_chars"] :]
        tail = body_previous[-60:].split(" ")[1:]  # a word-aligned slice of the tail
        assert " ".join(tail) in body_current


def test_overlap_can_be_disabled():
    description = "Missions :\n" + " ".join(f"mot{i}" for i in range(500))
    chunks = chunk_offer(make_offer(description), overlap=0)
    first_body = chunks[0]["text"][chunks[0]["header_chars"] :]
    second_body = chunks[1]["text"][chunks[1]["header_chars"] :]
    assert first_body.split()[-1] not in second_body.split()[:3]


def test_a_short_offer_is_one_chunk():
    chunks = chunk_offer(make_offer("Développeur Python en CDI à Lille. Stack FastAPI."))
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_id"].endswith(":0")


def test_a_truncated_offer_is_one_chunk_and_stays_flagged():
    # 63% of the corpus looks like this: ~500 characters cut mid-sentence.
    chunks = chunk_offer(make_offer("x" * 480, is_truncated=True))
    assert len(chunks) == 1
    assert chunks[0]["is_truncated"] is True


def test_an_empty_description_still_produces_a_findable_chunk():
    chunks = chunk_offer(make_offer(""))
    assert len(chunks) == 1
    assert chunks[0]["text"] == build_header(make_offer(""))
    assert chunks[0]["section"] is None


def test_sections_are_recorded_on_the_chunks():
    description = (
        "Missions :\n" + "Développer des API. " * 40 + "\n\nAvantages\n" + "Tickets. " * 40
    )
    chunks = chunk_offer(make_offer(description))
    labels = {chunk["section"] for chunk in chunks}
    assert "missions" in labels
    assert any("avantages" in chunk["sections"] for chunk in chunks)


def test_a_tiny_trailing_section_does_not_become_its_own_chunk():
    description = "Missions :\n" + "Développer des API. " * 50 + "\n\nAvantages\nTR."
    chunks = chunk_offer(make_offer(description), min_chars=200)
    assert all(chunk["n_chars"] > 150 for chunk in chunks)
    assert "TR." in chunks[-1]["text"]


def test_chunking_is_deterministic():
    offer = make_offer("Missions :\n" + "Développer. " * 200)
    assert chunk_offer(offer) == chunk_offer(offer)


def test_chunk_ids_are_stable_and_unique():
    chunks = chunk_offer(make_offer("Missions :\n" + "Développer. " * 200))
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == [f"{chunks[0]['offer_id']}:{i}" for i in range(len(chunks))]


@pytest.mark.parametrize("max_chars", [400, 800, 1200, 2000])
def test_the_budget_holds_for_every_configuration(max_chars):
    offers = [
        make_offer("Missions :\n" + "Développer des API REST. " * 120),
        make_offer("Un seul mot: " + "a" * 3000),
        make_offer("Court."),
    ]
    for offer in offers:
        for chunk in chunk_offer(offer, max_chars=max_chars, overlap=max_chars // 8):
            assert chunk["n_chars"] <= max_chars


def test_chunking_the_committed_sample():
    if not SAMPLE.exists():
        pytest.skip("sample not built: make sample")
    with SAMPLE.open(encoding="utf-8") as fh:
        offers = [json.loads(line) for line in fh]
    chunks = [chunk for offer in offers for chunk in chunk_offer(offer)]
    assert len(chunks) >= len(offers)
    assert all(chunk["n_chars"] <= DEFAULT_MAX_CHARS for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    # Every offer must be represented, truncated ones included.
    assert {chunk["offer_id"] for chunk in chunks} == {offer["id"] for offer in offers}
