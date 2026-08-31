"""Reference integrity checking.

Parses a LaTeX ``thebibliography`` block, resolves each entry against the
Crossref REST API, and flags the failure modes that indicate a broken or
fabricated citation.

The script **reports only**. It never edits the bibliography. Every non-clean
entry is emitted with a publisher landing-page URL so a human can confirm it in
one click -- the tool narrows the search, a person makes the call.

Four checks:

1. **Crossref resolution.** Best title match, then compare author surnames,
   volume, issue, pages, and year against what the entry claims.
2. **Duplicate titles**, matched case- and whitespace-insensitively. Two entries
   with the same title and disjoint author lists is the fabrication signature,
   and an exact-match comparison misses it when the only difference is
   ``wav2vec2`` vs ``Wav2Vec2``.
3. **Duplicate article coordinates** -- same venue, volume, issue, and pages --
   flagged independently of title. Two entries claiming the identical article
   slot cannot both be real regardless of what their titles say.
4. **Uncited entries.** A reference that appears in the bibliography but is
   never cited in the body is padding at best.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Reference",
    "CrossrefRecord",
    "Finding",
    "parse_bibliography",
    "parse_bibtex",
    "parse_citation_keys",
    "normalise_title",
    "surnames_from_authors",
    "find_duplicate_titles",
    "find_duplicate_coordinates",
    "audit",
    "render_report",
    "run_audit",
    "CrossrefClient",
    "best_crossref_match",
    "title_similarity",
    "TIER_CONFIRMED",
    "TIER_MANUAL",
    "TIER_FABRICATION",
]

CROSSREF_ENDPOINT = "https://api.crossref.org/works"

# A best-match title similarity at or above this is treated as the same work.
STRONG_MATCH = 0.90
# Between WEAK_MATCH and STRONG_MATCH the match is reported but not trusted.
WEAK_MATCH = 0.70


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """One ``\\bibitem`` as written in the .tex."""

    index: int  # 1-based; matches the [n] a reader sees
    key: str
    authors_raw: str
    surnames: Tuple[str, ...]
    title: str
    venue: str
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    year: Optional[int]
    raw: str
    doi: Optional[str] = None

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.key}"

    def coordinates(self) -> Optional[Tuple[str, str, str, str]]:
        """(venue, volume, issue, pages), normalised. None if underspecified.

        This is the article slot the entry claims to occupy. Two entries with
        the same slot are making incompatible claims.
        """
        if not self.venue or not self.volume or not self.pages:
            return None
        return (
            normalise_title(self.venue),
            _normalise_token(self.volume),
            _normalise_token(self.issue or ""),
            _normalise_token(self.pages),
        )


_BIB_BLOCK = re.compile(
    r"\\begin\{thebibliography\}.*?\n(?P<body>.*?)\\end\{thebibliography\}",
    re.DOTALL,
)
_BIBITEM = re.compile(r"\\bibitem\{(?P<key>[^}]+)\}", re.MULTILINE)
_TITLE = re.compile(r"``(?P<title>.+?),?''", re.DOTALL)
_VENUE = re.compile(r"\\textit\{(?P<venue>[^}]*)\}")
_VOLUME = re.compile(r"\bvol\.\s*([0-9A-Za-z\-]+)")
_ISSUE = re.compile(r"\bno\.\s*([0-9A-Za-z\-]+)")
_PAGES = re.compile(r"\bpp?\.\s*([^,]+?)\s*(?:,|\.$)")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def parse_bibliography(tex: str) -> List[Reference]:
    """Extract every ``\\bibitem`` from the thebibliography environment."""
    block = _BIB_BLOCK.search(tex)
    if not block:
        raise ValueError("no \\begin{thebibliography} ... \\end{thebibliography} found")

    body = block.group("body")
    matches = list(_BIBITEM.finditer(body))
    if not matches:
        raise ValueError("thebibliography contains no \\bibitem entries")

    references: List[Reference] = []
    for order, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[order].start() if order < len(matches) else len(body)
        chunk = body[start:end].strip()
        references.append(_parse_entry(order, match.group("key"), chunk))
    return references


_BIB_ENTRY = re.compile(
    r"@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,", re.MULTILINE
)


def parse_bibtex(bib: str) -> List[Reference]:
    """Extract citation metadata from a BibTeX database without editing it.

    The submission manuscript uses BibTeX rather than an inline
    ``thebibliography`` block. This deliberately small parser handles the
    balanced-brace records used by the project and keeps the audit free of a
    third-party parsing dependency.
    """
    references: List[Reference] = []
    for match in _BIB_ENTRY.finditer(bib):
        kind = match.group("kind").lower()
        if kind in {"comment", "preamble", "string"}:
            continue
        end = _bib_entry_end(bib, match.end())
        if end is None:
            raise ValueError(f"unterminated BibTeX entry: {match.group('key')}")
        fields = _bib_fields(bib[match.end():end - 1])
        authors_raw = _clean_latex(fields.get("author", ""))
        title = _clean_latex(fields.get("title", ""))
        venue = _clean_latex(
            fields.get("journal") or fields.get("booktitle") or fields.get("series", "")
        )
        year_match = re.search(r"(?:19|20)\d{2}", fields.get("year", ""))
        references.append(
            Reference(
                index=len(references) + 1,
                key=match.group("key"),
                authors_raw=authors_raw,
                surnames=_bib_surnames(authors_raw),
                title=title,
                venue=venue,
                volume=_clean_latex(fields["volume"]) if fields.get("volume") else None,
                issue=_clean_latex(fields["number"]) if fields.get("number") else None,
                pages=_clean_latex(fields["pages"]) if fields.get("pages") else None,
                year=int(year_match.group()) if year_match else None,
                raw=bib[match.start():end],
                doi=_clean_latex(fields["doi"]).lower() if fields.get("doi") else None,
            )
        )
    if not references:
        raise ValueError("BibTeX file contains no citation entries")
    return references


def _bib_entry_end(text: str, start: int) -> Optional[int]:
    """Return the position after the outer brace of a BibTeX entry."""
    depth, quoted, escaped = 1, False, False
    for position in range(start, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    return None


def _balanced_brace_end(text: str, start: int) -> Optional[int]:
    """Return the position after a brace-delimited BibTeX field value."""
    if start >= len(text) or text[start] != "{":
        return None
    depth, escaped = 0, False
    for position in range(start, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    return None


def _bib_fields(text: str) -> Dict[str, str]:
    """Read simple BibTeX field assignments, preserving nested braces."""
    fields: Dict[str, str] = {}
    pos, length = 0, len(text)
    while pos < length:
        while pos < length and text[pos] in "\t\r\n ,":
            pos += 1
        name = re.match(r"[A-Za-z][A-Za-z0-9_-]*", text[pos:])
        if not name:
            break
        key = name.group().lower()
        pos += len(name.group())
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length or text[pos] != "=":
            break
        pos += 1
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        if text[pos] == "{":
            end = _balanced_brace_end(text, pos)
            if end is None:
                break
            fields[key] = text[pos + 1:end - 1]
            pos = end
        elif text[pos] == '"':
            end = pos + 1
            escaped = False
            while end < length:
                if not escaped and text[end] == '"':
                    break
                escaped = (not escaped and text[end] == "\\")
                if text[end] != "\\":
                    escaped = False
                end += 1
            fields[key] = text[pos + 1:end]
            pos = end + 1
        else:
            end = text.find(",", pos)
            if end == -1:
                end = length
            fields[key] = text[pos:end].strip()
            pos = end + 1
    return fields


def _bib_surnames(authors_raw: str) -> Tuple[str, ...]:
    """BibTeX author values use ``Surname, Given and ...`` ordering."""
    surnames: List[str] = []
    for person in re.split(r"\s+and\s+", authors_raw, flags=re.IGNORECASE):
        person = person.strip()
        if not person:
            continue
        surname = person.split(",", 1)[0].strip() if "," in person else person.split()[-1]
        token = _normalise_token(surname)
        if token:
            surnames.append(token)
    return tuple(dict.fromkeys(surnames))


def _parse_entry(index: int, key: str, chunk: str) -> Reference:
    title_match = _TITLE.search(chunk)
    title = _clean_latex(title_match.group("title")) if title_match else ""

    # Authors are everything before the opening quote of the title.
    authors_raw = chunk[: title_match.start()] if title_match else chunk
    authors_raw = _clean_latex(authors_raw).strip().rstrip(",").strip()

    tail = chunk[title_match.end():] if title_match else chunk
    venue_match = _VENUE.search(tail)
    venue = _clean_latex(venue_match.group("venue")) if venue_match else ""

    volume = _first_group(_VOLUME, tail)
    issue = _first_group(_ISSUE, tail)
    pages = _first_group(_PAGES, tail)
    if pages:
        pages = _clean_latex(pages).strip().rstrip(".")

    years = _YEAR.findall(tail)
    year = None
    if years:
        # findall returns the group, not the match; re-scan for the full token.
        year_tokens = re.findall(r"\b((?:19|20)\d{2})\b", tail)
        year = int(year_tokens[-1])

    return Reference(
        index=index,
        key=key,
        authors_raw=authors_raw,
        surnames=surnames_from_authors(authors_raw),
        title=title,
        venue=venue,
        volume=volume,
        issue=issue,
        pages=pages,
        year=year,
        raw=chunk,
    )


def _first_group(pattern: re.Pattern, text: str) -> Optional[str]:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _clean_latex(text: str) -> str:
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    text = text.replace("--", "-").replace("~", " ").replace("\\&", "&")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


_CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{(?P<keys>[^}]+)\}")


def parse_citation_keys(tex: str) -> set[str]:
    """Keys cited in the body.

    The bibliography block is excised first, so a ``\\cite`` appearing inside a
    reference entry cannot make that entry look cited.
    """
    body = _BIB_BLOCK.sub("", tex)
    keys: set[str] = set()
    for match in _CITE.finditer(body):
        keys.update(part.strip() for part in match.group("keys").split(",") if part.strip())
    return keys


def _tex_tree(path: Path) -> str:
    """Read a manuscript root and its ``\\input`` files for citation auditing."""
    root = path.parent.resolve()
    seen: set[Path] = set()

    def visit(candidate: Path) -> str:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            return ""
        seen.add(candidate)
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        chunks = [text]
        for match in re.finditer(r"\\input\{([^}]+)\}", text):
            name = match.group(1)
            child = root / (name if Path(name).suffix else name + ".tex")
            chunks.append(visit(child))
        return "\n".join(chunks)

    return visit(path)


# --------------------------------------------------------------------------
# Normalisation and comparison
# --------------------------------------------------------------------------
def normalise_title(title: str) -> str:
    """Case-, accent-, punctuation- and whitespace-insensitive form.

    Case folding is the point: [16] differs from [6] only by ``wav2vec2`` vs
    ``Wav2Vec2``, which an exact comparison would miss.
    """
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def surnames_from_authors(authors_raw: str) -> Tuple[str, ...]:
    """Surnames from a written-out author list.

    Handles ``B. Schuller, B. Vlasenko, and A. Wendemuth`` by taking the last
    whitespace-separated token of each comma-separated part.
    """
    if not authors_raw:
        return ()
    text = re.sub(r"\band\b", ",", authors_raw)
    surnames: List[str] = []
    for part in text.split(","):
        part = part.strip().rstrip(".").strip()
        if not part:
            continue
        tokens = [t for t in part.split() if t]
        if not tokens:
            continue
        surname = tokens[-1].strip(".")
        # A bare initial ("B.") is not a surname.
        if len(surname) < 2:
            continue
        surnames.append(_normalise_token(surname))
    return tuple(dict.fromkeys(s for s in surnames if s))


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalise_title(left), normalise_title(right)).ratio()


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossrefRecord:
    doi: str
    title: str
    surnames: Tuple[str, ...]
    authors_display: str
    venue: str
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    year: Optional[int]
    similarity: float

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}"


class CrossrefClient:
    """Cached Crossref lookups.

    Responses are cached on disk keyed by normalised query, so re-running the
    audit is free and offline. ``allow_network=False`` makes it cache-only,
    which is what the tests use -- no test in this repo touches the network.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        mailto: Optional[str] = None,
        allow_network: bool = True,
        rows: int = 5,
        delay_seconds: float = 0.4,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.mailto = mailto
        self.allow_network = allow_network
        self.rows = rows
        self.delay_seconds = delay_seconds
        self._cache: Dict[str, Any] = {}
        self._last_request = 0.0
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                self._cache = json.load(handle)

    def search(self, title: str) -> List[Dict[str, Any]]:
        key = normalise_title(title)
        if key in self._cache:
            return self._cache[key]
        if not self.allow_network:
            return []

        params = {"query.bibliographic": title, "rows": str(self.rows)}
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{CROSSREF_ENDPOINT}?{urllib.parse.urlencode(params)}"

        # Crossref asks for a polite request rate.
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

        request = urllib.request.Request(
            url, headers={"User-Agent": f"ser-refs-audit/1.0 (mailto:{self.mailto or 'n/a'})"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._last_request = time.monotonic()

        items = payload.get("message", {}).get("items", [])
        self._cache[key] = items
        self._flush()
        return items

    def by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Resolve an explicit DOI before falling back to fuzzy title search."""
        clean = doi.removeprefix("https://doi.org/").lower()
        key = f"doi:{clean}"
        if key in self._cache:
            return self._cache[key]
        if not self.allow_network:
            return None
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        url = f"{CROSSREF_ENDPOINT}/{urllib.parse.quote(clean, safe='')}"
        request = urllib.request.Request(
            url, headers={"User-Agent": f"ser-refs-audit/1.0 (mailto:{self.mailto or 'n/a'})"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            item = json.loads(response.read().decode("utf-8")).get("message", {})
        self._last_request = time.monotonic()
        self._cache[key] = item
        self._flush()
        return item

    def _flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=1, sort_keys=True)


def best_crossref_match(reference: Reference, items: Sequence[Dict[str, Any]]) -> Optional[CrossrefRecord]:
    best: Optional[CrossrefRecord] = None
    for item in items:
        titles = item.get("title") or []
        if not titles:
            continue
        record = _to_record(item, title_similarity(reference.title, titles[0]))
        if best is None or record.similarity > best.similarity:
            best = record
    return best


def _to_record(item: Dict[str, Any], similarity: float) -> CrossrefRecord:
    authors = item.get("author") or []
    surnames = tuple(
        _normalise_token(a["family"]) for a in authors if a.get("family")
    )
    display = ", ".join(
        f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
        for a in authors
    )
    containers = item.get("container-title") or []
    dated = (item.get("published-print") or item.get("issued") or {}).get("date-parts") or [[None]]
    year = dated[0][0] if dated and dated[0] else None

    return CrossrefRecord(
        doi=item.get("DOI", ""),
        title=(item.get("title") or [""])[0],
        surnames=surnames,
        authors_display=display,
        venue=containers[0] if containers else "",
        volume=item.get("volume"),
        issue=item.get("issue"),
        pages=item.get("page") or item.get("article-number"),
        year=int(year) if isinstance(year, int) else None,
        similarity=similarity,
    )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def find_duplicate_titles(references: Sequence[Reference]) -> Dict[str, List[Reference]]:
    """Group references by normalised title, keeping groups of 2 or more."""
    groups: Dict[str, List[Reference]] = {}
    for reference in references:
        if not reference.title:
            continue
        groups.setdefault(normalise_title(reference.title), []).append(reference)
    return {key: group for key, group in groups.items() if len(group) > 1}


def find_duplicate_coordinates(
    references: Sequence[Reference],
) -> Dict[Tuple[str, str, str, str], List[Reference]]:
    """Group references claiming the same venue+volume+issue+pages."""
    groups: Dict[Tuple[str, str, str, str], List[Reference]] = {}
    for reference in references:
        coordinates = reference.coordinates()
        if coordinates is None:
            continue
        groups.setdefault(coordinates, []).append(reference)
    return {key: group for key, group in groups.items() if len(group) > 1}


TIER_CONFIRMED = "A. confirmed correct"
TIER_MANUAL = "B. needs manual resolution"
TIER_FABRICATION = "C. probable fabrication"


@dataclass
class Finding:
    reference: Reference
    match: Optional[CrossrefRecord]
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tier: str = TIER_CONFIRMED
    resolved: bool = False  # Crossref returned a confident match for this title

    @property
    def confirmed(self) -> bool:
        """Independently corroborated: right paper, right authors, right slot."""
        return self.resolved and not ({"AUTHOR-MISMATCH", "VOLUME-MISMATCH"} & set(self.flags))

    @property
    def status(self) -> str:
        return "VERIFIED" if not self.flags else " + ".join(self.flags)

    @property
    def landing_url(self) -> str:
        """Where a human should click to settle this entry.

        Only a *confident* title match yields a DOI. Linking the DOI of a
        different paper would send the reader somewhere irrelevant and make a
        wrong record look authoritative.
        """
        if self.resolved and self.match and self.match.doi:
            return self.match.url
        query = urllib.parse.quote_plus(self.reference.title[:200])
        return f"https://search.crossref.org/search/works?q={query}&from_ui=yes"


def audit(
    references: Sequence[Reference],
    cited_keys: Iterable[str],
    lookup: Callable[[Reference], Optional[CrossrefRecord]],
) -> List[Finding]:
    """Run every check and assign a tier to each reference.

    Ordering matters. Crossref resolution runs for every entry first, because
    the duplicate check needs to know which member of a duplicate pair is
    independently corroborated -- otherwise it condemns the real paper
    alongside the fabricated one.
    """
    cited = set(cited_keys)

    findings: List[Finding] = []
    for reference in references:
        match = lookup(reference)
        finding = Finding(reference=reference, match=match)
        _check_crossref(finding, reference, match)
        findings.append(finding)

    by_key = {finding.reference.key: finding for finding in findings}
    duplicate_titles = find_duplicate_titles(references)
    duplicate_coordinates = find_duplicate_coordinates(references)

    for finding in findings:
        _check_duplicates(
            finding, finding.reference, duplicate_titles, duplicate_coordinates, by_key
        )
        if finding.reference.key not in cited:
            finding.flags.append("UNCITED")
            finding.notes.append("Never cited in the body text.")
        finding.tier = _assign_tier(finding)

    return findings


def _check_crossref(
    finding: Finding, reference: Reference, match: Optional[CrossrefRecord]
) -> None:
    """Resolve against Crossref.

    Metadata is compared **only** when the title match is confident. Comparing
    authors and volumes against a record that is a different paper produces
    false accusations of fabrication -- the exact failure this tool exists to
    avoid making.
    """
    if match is None or match.similarity < STRONG_MATCH:
        finding.flags.append("NOT-IN-CROSSREF")
        if match is None:
            finding.notes.append("Crossref returned no candidate for this title.")
        else:
            finding.notes.append(
                f"Best Crossref candidate is a different paper "
                f"(title similarity {match.similarity:.2f}): "
                f"“{match.title}” by {match.authors_display or 'unknown'}. "
                "Metadata deliberately NOT compared against it."
            )
        finding.notes.append(
            "Absence from Crossref is **not** evidence of fabrication: NeurIPS, JMLR, "
            "Interspeech and arXiv-only work are frequently unindexed. This entry is "
            "simply unverifiable by this tool and needs a human."
        )
        return

    finding.resolved = True

    if reference.surnames and match.surnames:
        if not (set(reference.surnames) & set(match.surnames)):
            finding.flags.append("AUTHOR-MISMATCH")
            finding.notes.append(
                f"No surname overlap. Entry claims: {', '.join(reference.surnames)}. "
                f"Crossref has: {', '.join(match.surnames)}."
            )

    mismatches = []
    if reference.volume and match.volume and _normalise_token(reference.volume) != _normalise_token(match.volume):
        mismatches.append(f"volume {reference.volume} vs Crossref {match.volume}")
    if reference.pages and match.pages and not _pages_agree(reference.pages, match.pages):
        mismatches.append(f"pages {reference.pages} vs Crossref {match.pages}")
    if reference.year and match.year and reference.year != match.year:
        mismatches.append(f"year {reference.year} vs Crossref {match.year}")

    if mismatches:
        finding.flags.append("VOLUME-MISMATCH")
        finding.notes.append("; ".join(mismatches) + ".")


def _pages_agree(claimed: str, crossref: str) -> bool:
    left = _normalise_token(claimed)
    right = _normalise_token(crossref)
    return left == right or left in right or right in left


def _check_duplicates(
    finding: Finding,
    reference: Reference,
    duplicate_titles: Dict[str, List[Reference]],
    duplicate_coordinates: Dict[Tuple[str, str, str, str], List[Reference]],
    by_key: Dict[str, Finding],
) -> None:
    """Flag duplicate groups, distinguishing the real entry from the impostor.

    A duplicate pair is not symmetric. When one member is independently
    corroborated by Crossref and the other is not, the corroborated one is the
    real citation and only the other is suspect. Condemning both would make the
    report accuse a paper that demonstrably exists.
    """
    groups: List[Tuple[str, List[Reference]]] = []

    title_group = duplicate_titles.get(normalise_title(reference.title), [])
    if len(title_group) > 1:
        groups.append(("title", title_group))

    coordinates = reference.coordinates()
    coordinate_group = duplicate_coordinates.get(coordinates, []) if coordinates else []
    if len(coordinate_group) > 1:
        groups.append(("coordinates", coordinate_group))

    for kind, group in groups:
        others = [r for r in group if r.key != reference.key]
        description = (
            "Title is identical (case-insensitively) to "
            if kind == "title"
            else "Claims the same venue/volume/issue/pages as "
        )
        suffix = "." if kind == "title" else " -- two entries cannot occupy one article slot."
        finding.notes.append(description + ", ".join(r.label for r in others) + suffix)

        others_confirmed = [
            other for other in others if by_key[other.key].confirmed
        ]

        if finding.confirmed and not others_confirmed:
            # This entry is the real one; the duplicate is the problem.
            flag = "DUPLICATED-BY-OTHER"
            finding.notes.append(
                "This entry is independently confirmed by Crossref (authors, volume and "
                "pages all agree), so it is the genuine citation. The duplicate is the "
                "entry to remove."
            )
        elif others_confirmed:
            flag = "DUPLICATE-TITLE" if kind == "title" else "DUPLICATE-COORDINATES"
            finding.notes.append(
                "The duplicate "
                + ", ".join(o.label for o in others_confirmed)
                + " IS confirmed by Crossref while this entry is not."
            )
            for other in others_confirmed:
                if reference.surnames and other.surnames:
                    if not (set(reference.surnames) & set(other.surnames)):
                        finding.notes.append(
                            f"Author list disjoint from the confirmed {other.label} despite "
                            "the same work -- the fabrication signature."
                        )
        else:
            # Neither member corroborated; a human decides which (if either) is real.
            flag = "DUPLICATE-TITLE" if kind == "title" else "DUPLICATE-COORDINATES"
            finding.notes.append(
                "Neither entry in this duplicate group is confirmed by Crossref; both "
                "need manual resolution."
            )

        if flag not in finding.flags:
            finding.flags.append(flag)


def _assign_tier(finding: Finding) -> str:
    flags = set(finding.flags)

    # Duplicating a work that IS confirmed elsewhere in the list, with a
    # different author list, is not a mistake a real citation makes.
    if flags & {"DUPLICATE-TITLE", "DUPLICATE-COORDINATES"}:
        return TIER_FABRICATION
    if "AUTHOR-MISMATCH" in flags and flags & {"VOLUME-MISMATCH", "UNCITED"}:
        return TIER_FABRICATION

    # Being duplicated by someone else is not this entry's fault.
    substantive = flags - {"DUPLICATED-BY-OTHER"}
    if not substantive:
        return TIER_CONFIRMED
    return TIER_MANUAL


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
TIER_ORDER = (TIER_FABRICATION, TIER_MANUAL, TIER_CONFIRMED)


def render_report(findings: Sequence[Finding], *, source: str, offline: bool = False) -> str:
    lines: List[str] = []
    counts = {tier: sum(1 for f in findings if f.tier == tier) for tier in TIER_ORDER}

    lines.append("# Reference integrity report")
    lines.append("")
    lines.append(f"Source: `{source}` — {len(findings)} references.")
    lines.append("")
    lines.append(
        "Generated by `ser check-refs`. **This tool reports only; it never edits the "
        "bibliography.** Every entry in tiers B and C carries a landing-page link so "
        "each one can be settled by hand in a single click."
    )
    if offline:
        lines.append("")
        lines.append("> ⚠️ Run in cache-only mode; entries absent from the cache read as NOT-FOUND.")
    lines.append("")

    lines.append("| Tier | Count |")
    lines.append("|---|---|")
    for tier in TIER_ORDER:
        lines.append(f"| {tier} | {counts[tier]} |")
    lines.append("")

    actions = [(f, _action_for(f)) for f in findings if _action_for(f)]
    if actions:
        lines.append("## Actions")
        lines.append("")
        lines.append("| # | key | what to do | verify |")
        lines.append("|---|---|---|---|")
        for finding, action in actions:
            lines.append(
                f"| {finding.reference.index} | `{finding.reference.key}` | {action} | "
                f"[open]({finding.landing_url}) |"
            )
        lines.append("")
        lines.append(
            "Nothing here has been applied. Every change is yours to make by hand, "
            "after opening the link."
        )
        lines.append("")

    for tier in TIER_ORDER:
        group = [f for f in findings if f.tier == tier]
        if not group:
            continue
        lines.append(f"## {tier}")
        lines.append("")
        if tier == TIER_CONFIRMED:
            lines.append(
                "Crossref resolved the title, at least one author surname overlaps, and "
                "volume, pages, and year agree."
            )
            lines.append("")
            lines.append("| # | key | title | DOI | note |")
            lines.append("|---|---|---|---|---|")
            for finding in group:
                doi = f"[{finding.match.doi}]({finding.match.url})" if finding.match else "—"
                note = (
                    "**duplicated by another entry — that other entry is the one to delete**"
                    if "DUPLICATED-BY-OTHER" in finding.flags
                    else ""
                )
                lines.append(
                    f"| {finding.reference.index} | `{finding.reference.key}` | "
                    f"{_escape(finding.reference.title)} | {doi} | {note} |"
                )
            lines.append("")
            continue

        for finding in group:
            lines.extend(_render_detail(finding))

    lines.append("## All references")
    lines.append("")
    lines.append("| # | key | status | cited | tier |")
    lines.append("|---|---|---|---|---|")
    for finding in findings:
        cited = "no" if "UNCITED" in finding.flags else "yes"
        lines.append(
            f"| {finding.reference.index} | `{finding.reference.key}` | "
            f"{finding.status} | {cited} | {finding.tier[0]} |"
        )
    lines.append("")

    lines.append("## How to re-run")
    lines.append("")
    lines.append("```bash")
    lines.append("ser check-refs")
    lines.append("```")
    lines.append("")
    lines.append(
        "Crossref responses are cached, so re-runs are offline and free. "
        "The acceptance criterion for Phase 1 is that every entry resolves to tier A "
        "after the bibliography has been corrected **by hand**."
    )
    return "\n".join(lines) + "\n"


def _action_for(finding: Finding) -> str:
    """The single sentence a human needs in order to act on this entry."""
    flags = set(finding.flags)

    if flags & {"DUPLICATE-TITLE", "DUPLICATE-COORDINATES"}:
        return (
            "**Delete.** Duplicates a reference that Crossref confirms, under a "
            "different author list."
        )
    if "AUTHOR-MISMATCH" in flags and "VOLUME-MISMATCH" in flags:
        crossref = finding.match
        return (
            "**Correct the authors and volume** to the record at this DOI"
            + (f" ({crossref.authors_display}, vol. {crossref.volume})" if crossref else "")
            + "."
        )
    if "AUTHOR-MISMATCH" in flags:
        return "**Correct the author list** to match the DOI."
    if "VOLUME-MISMATCH" in flags:
        return "**Correct the volume, pages or year** to match the DOI."
    if "NOT-IN-CROSSREF" in flags:
        return (
            "Verify by hand. Not indexed in Crossref, which is normal for this venue "
            "and is not itself a problem."
        )
    if "UNCITED" in flags:
        return "Cite it in the body, or remove it."
    return ""


def _render_detail(finding: Finding) -> List[str]:
    reference = finding.reference
    match = finding.match
    lines = [f"### {reference.label}", ""]
    lines.append(f"**Status:** {finding.status}")
    lines.append("")
    lines.append("| | entry claims | Crossref |")
    lines.append("|---|---|---|")
    lines.append(
        f"| title | {_escape(reference.title)} | {_escape(match.title) if match else '—'} |"
    )
    lines.append(
        f"| authors | {_escape(reference.authors_raw)} | "
        f"{_escape(match.authors_display) if match else '—'} |"
    )
    lines.append(f"| venue | {_escape(reference.venue)} | {_escape(match.venue) if match else '—'} |")
    lines.append(
        f"| volume | {reference.volume or '—'} | {(match.volume if match else None) or '—'} |"
    )
    lines.append(
        f"| issue | {reference.issue or '—'} | {(match.issue if match else None) or '—'} |"
    )
    lines.append(
        f"| pages | {reference.pages or '—'} | {(match.pages if match else None) or '—'} |"
    )
    lines.append(f"| year | {reference.year or '—'} | {(match.year if match else None) or '—'} |")
    lines.append("")
    for note in finding.notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(f"**Verify:** {finding.landing_url}")
    lines.append("")
    return lines


def _escape(text: str) -> str:
    return (text or "").replace("|", "\\|")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_audit(
    tex_path: Path,
    out_path: Path,
    cache_path: Path,
    *,
    bib_path: Optional[Path] = None,
    mailto: Optional[str] = None,
    offline: bool = False,
) -> int:
    """Run the audit and write the report.

    Returns 0 only when every reference lands in tier A. A non-zero exit is what
    lets this gate a release check once the bibliography has been corrected --
    Phase 1's acceptance criterion is that a re-run comes back clean.
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        print(f"error: {tex_path} not found", file=sys.stderr)
        return 2

    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    if bib_path is None:
        references = parse_bibliography(tex)
        cited = parse_citation_keys(tex)
        source = tex_path.name
    else:
        bib_path = Path(bib_path)
        if not bib_path.exists():
            print(f"error: {bib_path} not found", file=sys.stderr)
            return 2
        references = parse_bibtex(bib_path.read_text(encoding="utf-8", errors="ignore"))
        cited = parse_citation_keys(_tex_tree(tex_path))
        source = f"{tex_path.name} + {bib_path.name}"

    client = CrossrefClient(cache_path, mailto=mailto, allow_network=not offline)

    def lookup(reference: Reference) -> Optional[CrossrefRecord]:
        if not reference.title:
            return None
        try:
            if reference.doi:
                item = client.by_doi(reference.doi)
                if item:
                    return _to_record(item, title_similarity(reference.title, (item.get("title") or [""])[0]))
            return best_crossref_match(reference, client.search(reference.title))
        except Exception as exc:  # network failures must not lose the whole run
            print(f"  ! Crossref lookup failed for {reference.label}: {exc}", file=sys.stderr)
            return None

    print(f"Parsed {len(references)} references from {bib_path.name if bib_path else tex_path.name}")
    print(f"Distinct keys cited in the body: {len(cited)}")

    findings = audit(references, cited, lookup)

    report = render_report(findings, source=source, offline=offline)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    counts = {tier: sum(1 for f in findings if f.tier == tier) for tier in TIER_ORDER}
    print()
    for tier in TIER_ORDER:
        print(f"  {tier:<28} {counts[tier]}")
    print()
    for finding in findings:
        if finding.tier != TIER_CONFIRMED:
            print(f"  {finding.reference.label:<28} {finding.status}")
    print()
    print(f"Report written to {out_path}")

    return 0 if counts[TIER_MANUAL] == 0 and counts[TIER_FABRICATION] == 0 else 1
