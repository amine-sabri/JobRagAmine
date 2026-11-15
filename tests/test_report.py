"""Marker injection: the mechanism that keeps the README honest."""

import pytest

from jobrag.report import MarkerError, inject_between_markers, markdown_table

DOC = """# Title

before

<!-- BEGIN: results -->
old content
<!-- END: results -->

after
"""


def test_injection_replaces_only_between_the_markers():
    out = inject_between_markers(DOC, "results", "new content")
    assert "old content" not in out
    assert "new content" in out
    assert out.startswith("# Title\n\nbefore\n")
    assert out.endswith("after\n")


def test_injection_is_idempotent():
    once = inject_between_markers(DOC, "results", "new content")
    assert inject_between_markers(once, "results", "new content") == once


def test_injection_keeps_the_markers_so_the_next_run_still_works():
    out = inject_between_markers(DOC, "results", "x")
    assert out.count("<!-- BEGIN: results -->") == 1
    assert out.count("<!-- END: results -->") == 1


def test_missing_markers_raise_instead_of_appending_silently():
    with pytest.raises(MarkerError):
        inject_between_markers("# No markers here\n", "results", "x")


def test_reversed_markers_raise():
    reversed_doc = "<!-- END: results -->\n<!-- BEGIN: results -->\n"
    with pytest.raises(MarkerError):
        inject_between_markers(reversed_doc, "results", "x")


def test_markdown_table_shape():
    table = markdown_table(["a", "b"], [[1, None], ["x", 2.5]])
    assert table.splitlines() == [
        "| a | b |",
        "| --- | --- |",
        "| 1 |  |",
        "| x | 2.5 |",
    ]
