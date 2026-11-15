"""Turning a source row into an export record.

Kept in the library rather than in the script because this is where the privacy
rules live: the column allow-list, the exclusions, and the cleaning applied to
every description. It is tested directly.
"""

import re
from typing import Any

from .clean import clean_description, content_hash, normalise_company, normalise_title
from .enums import CONTRACT_LABELS, LANGUAGE_CODES, SOURCE_NAMES, decode

# Exported. Everything else is deliberately left behind.
COLUMNS = [
    "Id",
    "Source",
    "ExternalId",
    "Title",
    "CompanyName",
    "Location",
    "City",
    "Region",
    "Country",
    "IsRemote",
    "ContractType",
    "Language",
    "Level",
    "Status",
    "Description",
    "Url",
    "Salary",
    "PostedAt",
    "DiscoveredAt",
    "ExpiresAt",
    "EndClientName",
    "PostedByIntermediary",
]

# Never exported: personal contact details, and the platform's own CV-matching
# columns, which say more about the person searching than about the offer.
EXCLUDED = [
    "ApplyEmail",
    "ContactName",
    "ApplyUrl",
    "BestScore",
    "BestCvProfileId",
    "CompanyId",
    "EndClientCompanyId",
    "Fingerprint",
    "CreatedAt",
    "UpdatedAt",
    "EmployerCheckedAt",
]

# The boards that truncate end the text with an ellipsis, and Adzuna cuts at
# exactly 500 characters.
TRUNCATED_RE = re.compile(r"(?:…|\.\.\.)\s*$")
TRUNCATION_LENGTH = 500


def is_truncated(raw_description: str) -> bool:
    """Did the source board cut this description short?"""
    if not raw_description:
        return False
    return bool(TRUNCATED_RE.search(raw_description)) or len(raw_description) == TRUNCATION_LENGTH


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def build_offer(row: dict[str, Any]) -> dict[str, Any]:
    """One database row -> one export record, cleaned and masked."""
    raw_description = row["Description"] or ""
    description, n_emails, n_phones = clean_description(raw_description)
    title = row["Title"] or ""
    company = row["CompanyName"] or ""
    return {
        "id": str(row["Id"]),
        "source": row["Source"],
        "source_name": SOURCE_NAMES.get(row["Source"], f"source-{row['Source']}"),
        "external_id": row["ExternalId"],
        "title": title,
        "title_norm": normalise_title(title),
        "company": company,
        "company_norm": normalise_company(company),
        "location": row["Location"],
        "city": row["City"],
        "region": row["Region"],
        "country": row["Country"],
        "is_remote": row["IsRemote"],
        "contract_type": row["ContractType"],
        "contract_label": decode(CONTRACT_LABELS, row["ContractType"], "contract"),
        "language": row["Language"],
        "language_code": LANGUAGE_CODES.get(row["Language"], f"lang-{row['Language']}"),
        # Status and Level are the platform's own pipeline states: kept raw,
        # undecoded, until their meaning is confirmed.
        "status": row["Status"],
        "level": row["Level"],
        "salary": row["Salary"],
        "url": row["Url"],
        "posted_at": _iso(row["PostedAt"]),
        "discovered_at": _iso(row["DiscoveredAt"]),
        "expires_at": _iso(row["ExpiresAt"]),
        "end_client_name": row["EndClientName"],
        "posted_by_intermediary": row["PostedByIntermediary"],
        "description": description,
        "description_chars": len(description),
        # A truncated offer can be cited for its metadata, but its text does not
        # contain the requirements.
        "is_truncated": is_truncated(raw_description),
        "description_hash": content_hash(description),
        "masked_emails": n_emails,
        "masked_phones": n_phones,
    }
