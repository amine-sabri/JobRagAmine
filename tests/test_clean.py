"""Masking and normalisation. Expected values are written by hand, not generated."""

import pytest

from jobrag.clean import (
    clean_description,
    content_hash,
    html_to_text,
    mask_contacts,
    normalise_company,
    normalise_title,
    normalise_whitespace,
)


@pytest.mark.parametrize(
    "raw",
    [
        "jean.dupont@acme.fr",
        "Jean-Luc.Dupont+rh@acme-corp.co.uk",
        "r.h@sous.domaine.example.com",
        "CANDIDATURE@ENTREPRISE.FR",
    ],
)
def test_emails_are_masked(raw):
    masked, n_emails, n_phones = mask_contacts(f"Envoyez votre CV à {raw} avant lundi.")
    assert raw not in masked
    assert masked == "Envoyez votre CV à [EMAIL] avant lundi."
    assert (n_emails, n_phones) == (1, 0)


@pytest.mark.parametrize(
    "raw",
    [
        "06 12 34 56 78",
        "06.12.34.56.78",
        "06-12-34-56-78",
        "0612345678",
        "+33 6 12 34 56 78",
        "+33(0)6 12 34 56 78",
        "0033612345678",
        "01 23 45 67 89",
    ],
)
def test_phone_numbers_are_masked(raw):
    masked, n_emails, n_phones = mask_contacts(f"Contactez-nous au {raw} pour postuler.")
    assert masked == "Contactez-nous au [PHONE] pour postuler."
    assert (n_emails, n_phones) == (0, 1)


@pytest.mark.parametrize(
    "raw",
    [
        # An ISO timestamp used to match: "02.190606-04" looks like a phone number.
        "Publiée le 2026-10-10T01:16:02.190606-04:00",
        "SIRET 01234567890123",
        "Budget 500 000 euros",
        "Référence 2024-2025 du poste",
        "Version 1.2.3.4.5 du logiciel",
        "Entre 45 000 et 55 000 euros",
    ],
)
def test_non_phone_numbers_survive(raw):
    masked, n_emails, n_phones = mask_contacts(raw)
    assert masked == raw
    assert (n_emails, n_phones) == (0, 0)


def test_masking_counts_every_occurrence_and_is_idempotent():
    text = "rh@acme.fr ou recrutement@acme.fr, tél 06 12 34 56 78 ou 07.11.22.33.44"
    once, n_emails, n_phones = mask_contacts(text)
    assert (n_emails, n_phones) == (2, 2)
    assert once == "[EMAIL] ou [EMAIL], tél [PHONE] ou [PHONE]"
    twice, n_emails, n_phones = mask_contacts(once)
    assert twice == once and (n_emails, n_phones) == (0, 0)


def test_mask_contacts_handles_empty():
    assert mask_contacts("") == ("", 0, 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>Missions&nbsp;:</p><ul><li>Python</li></ul>", "Missions :\nPython"),
        ("Pas de balise ici", "Pas de balise ici"),
        ("Ligne 1<br/>Ligne 2", "Ligne 1\nLigne 2"),
        ("Salaire &gt; 45k&euro;", "Salaire > 45k€"),
        ("<div><strong>Profil</strong> recherché</div>", "Profil recherché"),
    ],
)
def test_html_is_unwrapped(raw, expected):
    assert normalise_whitespace(html_to_text(raw)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  deux   espaces  ", "deux espaces"),
        ("espace insécable", "espace insécable"),
        # A paragraph break survives: phase 2 needs it to find sections.
        ("Missions\n\nProfil", "Missions\n\nProfil"),
        ("trop\n\n\n\nde lignes", "trop\n\nde lignes"),
        ("\n  bordures  \n", "bordures"),
    ],
)
def test_whitespace_normalisation(raw, expected):
    assert normalise_whitespace(raw) == expected


def test_clean_description_runs_the_whole_pipeline():
    raw = "<p>Missions&nbsp;:</p><p>Contact : rh@acme.fr   au 06 12 34 56 78</p>"
    text, n_emails, n_phones = clean_description(raw)
    assert text == "Missions :\nContact : [EMAIL] au [PHONE]"
    assert (n_emails, n_phones) == (1, 1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Développeur Web Fullstack (H/F)", "developpeur web fullstack"),
        ("Développeur web fullstack H/F", "developpeur web fullstack"),
        ("Employé Polyvalent (F/H)", "employe polyvalent"),
        ("Data Engineer (h/f/d)", "data engineer"),
        ("03 AI Solution Engineer CDI · Paris", "03 ai solution engineer cdi"),
        ("Ingénieur C++ / Python", "ingenieur c++ python"),
        ("Dév. back-end  (M/F)", "dev back end"),
        ("", ""),
    ],
)
def test_title_normalisation(raw, expected):
    assert normalise_title(raw) == expected


def test_title_normalisation_is_idempotent():
    once = normalise_title("Développeur Web Fullstack (H/F) · Lille")
    assert normalise_title(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MEGAGENCE SAS", "megagence"),
        ("Crédit Agricole", "credit agricole"),
        ("ACME S.A.R.L.", "acme"),
        ("Thales SA", "thales"),
        ("H&M", "hm"),
        ("I.B.M. France", "ibm france"),
        ("", ""),
    ],
)
def test_company_normalisation(raw, expected):
    assert normalise_company(raw) == expected


def test_content_hash_is_stable_and_separates_fields():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    # Without a separator, ("ab", "c") and ("a", "bc") would collide.
    assert content_hash("ab", "c") != content_hash("a", "bc")
    assert len(content_hash("abc")) == 32
