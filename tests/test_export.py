"""The export record: what it contains, and above all what it must not."""

from datetime import UTC, datetime

import pytest

from jobrag.export import COLUMNS, EXCLUDED, build_offer, is_truncated

ROW = {
    "Id": "3f1b2c4d-0000-4000-8000-000000000001",
    "Source": 1,
    "ExternalId": "ft-12345",
    "Title": "Développeur Python (H/F)",
    "CompanyName": "ACME SAS",
    "Location": "Lille (59)",
    "City": "Lille",
    "Region": "Hauts-de-France",
    "Country": "FR",
    "IsRemote": True,
    "ContractType": 1,
    "Language": 0,
    "Level": 4,
    "Status": 2,
    "Description": "Missions : API FastAPI.\nContact : rh@acme.fr au 06 12 34 56 78",
    "Url": "https://candidat.francetravail.fr/offres/recherche/detail/12345",
    "Salary": "45 000 - 55 000 EUR",
    "PostedAt": datetime(2026, 10, 1, 8, 30, tzinfo=UTC),
    "DiscoveredAt": datetime(2026, 10, 2, 6, 0, tzinfo=UTC),
    "ExpiresAt": None,
    "EndClientName": None,
    "PostedByIntermediary": False,
}


def test_build_offer_decodes_and_cleans():
    offer = build_offer(ROW)
    assert offer["id"] == ROW["Id"]
    assert offer["source_name"] == "france-travail"
    assert offer["contract_label"] == "CDI"
    assert offer["language_code"] == "fr"
    assert offer["title_norm"] == "developpeur python"
    assert offer["company_norm"] == "acme"
    assert offer["posted_at"] == "2026-10-01T08:30:00+00:00"
    assert offer["expires_at"] is None
    assert offer["description_chars"] == len(offer["description"])


def test_build_offer_masks_contacts_and_counts_them():
    offer = build_offer(ROW)
    assert "rh@acme.fr" not in offer["description"]
    assert "06 12 34 56 78" not in offer["description"]
    assert offer["description"].endswith("Contact : [EMAIL] au [PHONE]")
    assert (offer["masked_emails"], offer["masked_phones"]) == (1, 1)


def test_the_record_carries_no_excluded_field():
    offer = build_offer(ROW)
    lowered = {key.lower() for key in offer}
    for column in EXCLUDED:
        assert column.lower() not in lowered, f"{column} leaked into the export"
    assert not set(COLUMNS) & set(EXCLUDED)


def test_build_offer_needs_exactly_the_selected_columns():
    # Keeps the SQL allow-list and the record builder in sync: a column removed
    # from COLUMNS without touching build_offer fails here, not in production.
    build_offer(dict.fromkeys(COLUMNS, None) | {"Id": "x", "Source": 1, "Language": 0})
    incomplete = dict(ROW)
    del incomplete["City"]
    with pytest.raises(KeyError):
        build_offer(incomplete)


def test_status_and_level_stay_raw_integers():
    # Their meaning is unconfirmed: nothing may depend on a guessed label.
    offer = build_offer(ROW)
    assert offer["status"] == 2
    assert offer["level"] == 4
    assert "status_label" not in offer
    assert "level_label" not in offer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Texte complet terminé normalement.", False),
        ("Un début de description…", True),
        ("Un début de description...", True),
        ("Un début de description… ", True),
        ("x" * 500, True),  # Adzuna cuts at exactly 500 characters
        ("x" * 499, False),
        ("", False),
    ],
)
def test_truncation_detection(raw, expected):
    assert is_truncated(raw) is expected


def test_unknown_enum_values_degrade_to_a_readable_label():
    offer = build_offer(dict(ROW) | {"Source": 99, "ContractType": 42, "Language": 7})
    assert offer["source_name"] == "source-99"
    assert offer["contract_label"] == "contract-42"
    assert offer["language_code"] == "lang-7"
