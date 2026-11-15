"""Enum decoding, including the values that actually occur in the corpus."""

import pytest

from jobrag.enums import (
    CONTRACT_LABELS,
    LANGUAGE_CODES,
    SOURCE_NAMES,
    STATUS_LABELS,
    ContractType,
    Language,
    decode,
)

# Measured with `select "ContractType", count(*) ... group by 1` on 32,013 offers.
OBSERVED_CONTRACT_TYPES = [0, 1, 2, 5, 6, 7]
OBSERVED_SOURCES = [0, 1, 2, 10, 12, 13, 14, 20, 50, 51]


@pytest.mark.parametrize("value", OBSERVED_CONTRACT_TYPES)
def test_every_observed_contract_type_has_a_label(value):
    assert CONTRACT_LABELS[value]
    assert not CONTRACT_LABELS[value].startswith("contract-")


@pytest.mark.parametrize("value", OBSERVED_SOURCES)
def test_every_observed_source_has_a_name(value):
    assert SOURCE_NAMES[value]


def test_language_codes_cover_both_observed_values():
    assert LANGUAGE_CODES[Language.FR] == "fr"
    assert LANGUAGE_CODES[Language.EN] == "en"


def test_contract_labels_cover_the_whole_enum():
    assert set(CONTRACT_LABELS) == {member.value for member in ContractType}


def test_decode_falls_back_readably():
    assert decode(CONTRACT_LABELS, 1, "contract") == "CDI"
    assert decode(CONTRACT_LABELS, 42, "contract") == "contract-42"
    assert decode(CONTRACT_LABELS, None, "contract") is None
    # Status has no verified mapping: decoding must say so rather than invent.
    assert decode(STATUS_LABELS, 2, "status") == "status-2"
