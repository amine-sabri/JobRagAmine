"""Filter extraction: the four guards against silent over-filtering."""

from datetime import date

import pytest

from jobrag.filters import Filters, extract_filters

CITIES = {"Lille", "Paris", "Lyon", "Saint-Étienne", "Nantes", "LILLE"}
TODAY = date(2026, 11, 16)


def extract(question: str):
    return extract_filters(question, cities=CITIES, today=TODAY)


@pytest.mark.parametrize(
    ("question", "contract"),
    [
        ("Quelles offres en CDI ?", 1),
        ("offres en CDD à Lyon", 2),
        ("un stage de fin d'études", 3),
        ("offres en alternance", 4),
        ("missions freelance", 5),
        ("offres en intérim", 6),
        ("offres de doctorat", 7),
        ("une thèse en informatique", 7),
        ("offres de développeur", None),
    ],
)
def test_contract_extraction(question, contract):
    assert extract(question).filters.contract_type == contract


def test_a_city_needs_a_locative_preposition():
    # "à Lille" is a place; "chez Nantes Métropole Habitat" is an employer, and
    # filtering on city=Nantes would drop that employer's other postings.
    assert extract("Quelles offres à Lille ?").filters.city in {"Lille", "LILLE"}
    assert extract("Quelles offres chez Nantes Métropole Habitat ?").filters.city is None


def test_two_cities_disable_the_city_filter():
    extraction = extract("des offres à Lille ou à Paris ?")
    assert extraction.filters.city is None
    assert any("several cities" in note for note in extraction.notes)


def test_an_unknown_city_is_not_invented():
    assert extract("Quelles offres à Reykjavik ?").filters.city is None


def test_a_negated_remote_mention_sets_no_filter():
    extraction = extract("offres sans télétravail à Lyon")
    assert extraction.filters.is_remote is None
    assert extraction.filters.city == "Lyon"
    assert any("negated" in note for note in extraction.notes)


def test_remote_is_detected_in_both_languages():
    assert extract("offres en télétravail").filters.is_remote is True
    assert extract("remote jobs").filters.is_remote is True
    assert extract("offres à distance").filters.is_remote is True


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("offres publiées aujourd'hui", date(2026, 11, 15)),
        ("offres de cette semaine", date(2026, 11, 9)),
        ("offres depuis 3 jours", date(2026, 11, 13)),
        ("offres de ce mois", date(2026, 10, 17)),
        ("offres récentes", date(2026, 10, 17)),
        ("offres de développeur", None),
    ],
)
def test_recency_extraction(question, expected):
    assert extract(question).filters.posted_after == expected


def test_language_extraction():
    assert extract("offres rédigées en anglais").filters.language == "en"
    assert extract("offres en français").filters.language == "fr"
    assert extract("offres de data engineer").filters.language is None


def test_filters_build_parameterised_sql():
    filters = Filters(city="Lille", contract_type=1, is_remote=True, posted_after=TODAY)
    conditions, params = filters.as_conditions()
    assert len(conditions) == 4
    # Values travel as bound parameters, never interpolated into the SQL.
    assert all("%(" in condition for condition in conditions)
    assert params == {
        "f_city": "Lille",
        "f_contract": 1,
        "f_remote": True,
        "f_posted_after": TODAY,
    }


def test_empty_filters_produce_no_sql():
    assert Filters().as_conditions() == ([], {})
    assert Filters().is_empty() is True
    assert Filters(city="Lille").is_empty() is False


def test_describe_is_human_readable():
    assert Filters().describe() == "aucun"
    assert "ville=Lille" in Filters(city="Lille").describe()
    assert "télétravail=non" in Filters(is_remote=False).describe()
