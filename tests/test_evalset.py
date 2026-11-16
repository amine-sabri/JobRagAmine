"""The evaluation-set schema, the labelling search, and the coverage accounting."""

from collections import Counter

import pytest

from jobrag.evalset import (
    TARGET_MIX,
    OfferSearch,
    content_tokens,
    coverage,
    describe_offer,
    document_frequencies,
    fold,
    guess_type,
    idf,
    load_questions,
    make_question,
    matches_filters,
    next_question_id,
    parse_query,
    save_questions,
    score_offer,
)


def offer(**fields):
    base = {
        "id": "o1",
        "title": "Développeur Python",
        "company": "ACME",
        "city": "Lille",
        "contract_type": 1,
        "contract_label": "CDI",
        "language_code": "fr",
        "is_remote": False,
        "is_truncated": False,
        "source_name": "france-travail",
        "description": "Nous cherchons un développeur Python avec Kubernetes.",
        "posted_at": "2026-10-01T00:00:00+00:00",
        "duplicate_count": 0,
    }
    return base | fields


def test_folding_removes_accents_and_case():
    assert fold("Développeur Sénior") == "developpeur senior"
    assert fold("") == ""


def test_content_tokens_drop_stopwords_and_noise():
    tokens = content_tokens("Quelles offres demandent du Kubernetes à Paris ?")
    assert "kubernetes" in tokens
    assert "paris" in tokens
    for noise in ("quelles", "offres", "demandent", "du", "a"):
        assert noise not in tokens


@pytest.mark.parametrize(
    ("raw", "text", "filters"),
    [
        ("kubernetes city:Lille", "kubernetes", {"city": "lille"}),
        ("city:Paris contract:CDI", "", {"city": "paris", "contract": "cdi"}),
        ("python CITY:Lyon", "python", {"city": "lyon"}),
        ("remote:true data", "data", {"remote": "true"}),
        ("pas de filtre", "pas de filtre", {}),
    ],
)
def test_parse_query(raw, text, filters):
    assert parse_query(raw) == (text, filters)


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"city": "lille"}, True),
        ({"city": "Lille"}, True),
        ({"city": "lyon"}, False),
        ({"contract": "1"}, True),
        ({"contract": "cdi"}, True),
        ({"contract": "2"}, False),
        ({"lang": "fr"}, True),
        ({"lang": "en"}, False),
        ({"remote": "true"}, False),
        ({"remote": "false"}, True),
        ({"source": "france"}, True),
        ({"truncated": "false"}, True),
        ({"city": "lille", "contract": "cdi"}, True),
        ({"city": "lille", "contract": "cdd"}, False),
        ({"unknown": "x"}, True),  # unknown keys are ignored, not fatal
    ],
)
def test_matches_filters(filters, expected):
    assert matches_filters(offer(), filters) is expected


def test_idf_ranks_rare_tokens_higher():
    frequencies = Counter({"paris": 5000, "kubernetes": 200})
    assert idf(frequencies, 23715, "kubernetes") > idf(frequencies, 23715, "paris")
    # An unseen token gets the highest weight, not a crash.
    assert idf(frequencies, 23715, "plomberiequantique") > idf(frequencies, 23715, "kubernetes")


def test_document_frequencies_count_offers_not_occurrences():
    frequencies = document_frequencies(["python python python", "python java"])
    assert frequencies["python"] == 2
    assert frequencies["java"] == 1


def test_score_is_zero_when_nothing_matches():
    target = offer(_folded=fold("Nous cherchons un développeur Python avec Kubernetes."))
    assert score_offer(target, ["plomberie"], {"plomberie": 5.0}) == 0


def test_a_title_hit_outscores_a_body_hit():
    weights = {"python": 2.0}
    in_title = offer(_folded=fold("texte sans le mot"), title="Développeur Python")
    in_body = offer(_folded=fold("un poste de python"), title="Développeur")
    assert score_offer(in_title, ["python"], weights) > score_offer(in_body, ["python"], weights)


def test_a_rare_token_outweighs_two_common_ones():
    # The ranking bug this fixed: two common words in a title beat the one rare
    # word the question is really about.
    weights = {"kubernetes": 5.0, "paris": 1.2, "cdi": 1.1}
    rare = offer(title="SRE Kubernetes", _folded=fold("SRE Kubernetes"))
    common = offer(title="Dev Paris CDI", _folded=fold("Dev Paris CDI"))
    assert score_offer(rare, ["kubernetes"], weights) > 0
    assert score_offer(rare, ["kubernetes", "paris", "cdi"], weights) > score_offer(
        common, ["kubernetes", "paris", "cdi"], weights
    )


@pytest.fixture
def searcher():
    return OfferSearch(
        [
            offer(id="a", title="SRE Kubernetes", description="Orchestration.", city="Paris"),
            offer(id="b", title="Développeur Paris CDI", description="JavaScript.", city="Paris"),
            offer(
                id="c", title="Data Engineer", description="Python, Spark, Kubernetes.", city="Lyon"
            ),
            offer(
                id="d",
                title="Plombier",
                description="Tuyaux.",
                city="Lyon",
                contract_type=2,
                contract_label="CDD",
            ),
        ]
    )


def test_search_ranks_the_rare_token_first(searcher):
    hits = searcher.search("kubernetes")
    assert [hit["id"] for hit in hits] == ["a", "c"]


def test_search_applies_filters(searcher):
    assert [hit["id"] for hit in searcher.search("kubernetes city:Lyon")] == ["c"]
    assert searcher.search("kubernetes city:Nantes") == []
    assert [hit["id"] for hit in searcher.search("tuyaux contract:CDD")] == ["d"]


def test_search_ignores_terms_already_given_as_filters(searcher):
    # "paris" as text would otherwise outrank the Kubernetes offer inside the
    # Paris subset — the exact bug this behaviour fixes.
    hits = searcher.search("kubernetes à Paris en CDI city:Paris contract:CDI")
    assert [hit["id"] for hit in hits] == ["a"]


def test_search_respects_the_limit(searcher):
    assert len(searcher.search("kubernetes", limit=1)) == 1


def test_an_empty_query_with_a_filter_lists_the_matching_offers(searcher):
    assert {hit["id"] for hit in searcher.search("city:Lyon")} == {"c", "d"}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Combien d'offres en CDI à Lille ?", "aggregation"),
        ("Quelles sont les compétences les plus demandées ?", "aggregation"),
        ("Pour cette offre, que me manque-t-il ?", "offer"),
        ("Quelles offres en CDI à Lyon ?", "filter"),
        ("Quelle entreprise recrute un développeur Python ?", "fact"),
    ],
)
def test_guess_type(question, expected):
    assert guess_type(question) == expected


def test_guess_type_uses_the_filters_as_a_hint():
    assert guess_type("offres kubernetes", {"city": "lille"}) == "filter"


def test_next_question_id():
    assert next_question_id([]) == "q001"
    assert next_question_id([{"id": "q001"}, {"id": "q002"}]) == "q003"
    # A gap must not reuse an id already taken.
    assert next_question_id([{"id": "q001"}, {"id": "q009"}]) == "q010"
    assert next_question_id([{"id": "oops"}]) == "q001"


def test_make_question_normalises_and_deduplicates():
    question = make_question("q001", "  Une question ?  ", "fact", ["a", "b", "a"])
    assert question["question"] == "Une question ?"
    assert question["relevant_offer_ids"] == ["a", "b"]
    assert question["type"] == "fact"
    assert question["created_at"]


def test_make_question_rejects_contradictions():
    with pytest.raises(ValueError, match="no relevant offer"):
        make_question("q001", "Sans réponse ?", "none", ["a"])
    with pytest.raises(ValueError, match="at least one relevant offer"):
        make_question("q001", "Avec réponse ?", "fact", [])
    with pytest.raises(ValueError, match="unknown type"):
        make_question("q001", "Question ?", "autre", ["a"])


def test_questions_round_trip_through_yaml(tmp_path):
    path = tmp_path / "questions.yaml"
    assert load_questions(path) == []  # missing file is an empty set, not an error
    questions = [
        make_question("q001", "Offres à Lille en CDI ?", "filter", ["a"]),
        make_question("q002", "Salaire médian en plomberie quantique ?", "none", []),
    ]
    save_questions(path, questions)
    assert load_questions(path) == questions
    # Accents must survive the round trip, not become \\uXXXX escapes.
    assert "médian" in path.read_text(encoding="utf-8")


def test_saving_is_atomic(tmp_path):
    path = tmp_path / "questions.yaml"
    save_questions(path, [make_question("q001", "Q ?", "fact", ["a"])])
    assert not list(tmp_path.glob("*.tmp"))


def test_loading_a_malformed_file_fails_loudly(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("question: pas une liste\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list of questions"):
        load_questions(path)


def test_coverage_counts_against_the_target_mix():
    questions = [
        make_question("q001", "a ?", "fact", ["x"]),
        make_question("q002", "b ?", "fact", ["x"]),
        make_question("q003", "c ?", "none", []),
    ]
    rows = coverage(questions)
    assert rows["fact"] == {
        "written": 2,
        "target": TARGET_MIX["fact"],
        "missing": TARGET_MIX["fact"] - 2,
    }
    assert rows["none"]["written"] == 1
    assert rows["offer"]["written"] == 0
    assert sum(TARGET_MIX.values()) == 100


def test_describe_offer_shows_what_a_labeller_needs():
    line = describe_offer(offer(is_truncated=True, duplicate_count=3))
    assert "Développeur Python" in line
    assert "ACME" in line and "Lille" in line and "CDI" in line
    assert "TRONQUÉE" in line
    assert "+3 copies" in line
    # Filler must not be printed as if it were information.
    assert "non précisé" not in describe_offer(offer(contract_label="non précisé"))
