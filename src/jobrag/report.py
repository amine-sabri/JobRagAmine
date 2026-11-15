"""Injecting generated output into Markdown files.

Every results table in README.md is produced by a script and written between
markers, so a number can always be traced back to the command that produced it.
"""

from pathlib import Path

BEGIN = "<!-- BEGIN: {name} -->"
END = "<!-- END: {name} -->"


class MarkerError(ValueError):
    """The target file does not contain the expected markers."""


def inject_between_markers(text: str, name: str, content: str) -> str:
    """Replace whatever sits between the markers named `name` with `content`.

    Idempotent: injecting the same content twice yields the same file.
    """
    begin, end = BEGIN.format(name=name), END.format(name=name)
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1:
        raise MarkerError(f"markers for {name!r} not found")
    if stop < start:
        raise MarkerError(f"markers for {name!r} are in the wrong order")
    return text[: start + len(begin)] + "\n" + content.strip() + "\n" + text[stop:]


def inject_into_file(path: Path, name: str, content: str) -> None:
    path.write_text(
        inject_between_markers(path.read_text(encoding="utf-8"), name, content), encoding="utf-8"
    )


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """A GitHub-flavoured table. Numbers are right-aligned by convention."""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)
