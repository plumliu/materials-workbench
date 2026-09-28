from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


# A suffix never adds hierarchy depth.  Bare suffix letters must be attached to
# the numeric core; allowing whitespace there would turn the first letter of an
# ordinary title into a suffix.
IDENTIFIER_TOKEN_PATTERN = (
    r"[1-9]\d*(?:\s*\.\s*\d+)*"
    r"(?:\s*\(\s*(?:[A-Za-z]|\d+|[ivxlcdmIVXLCDM]+)\s*\)"
    r"|\s*\[\s*(?:[A-Za-z]|\d+|[ivxlcdmIVXLCDM]+)\s*\]"
    r"|[.\-–—][A-Za-z]"
    r"|[A-Za-z](?=\s|$|\(|\[|[:\-–—]))*"
)

_PREFIX_RE = re.compile(
    rf"^\s*(?P<id>{IDENTIFIER_TOKEN_PATTERN})(?:\.(?=\s|\[|$))?(?=\s|\[|$|[:\-–—])"
)
_CORE_RE = re.compile(r"^[1-9]\d*(?:\s*\.\s*\d+)*")
_SUFFIX_RE = re.compile(
    r"\(\s*(?P<paren>[A-Za-z]|\d+|[ivxlcdmIVXLCDM]+)\s*\)"
    r"|\[\s*(?P<bracket>[A-Za-z]|\d+|[ivxlcdmIVXLCDM]+)\s*\]"
    r"|[.\-–—](?P<separated>[A-Za-z])"
    r"|(?P<bare>[A-Za-z])"
)
_CATALOG_KIND_RE = re.compile(r"^\s*\[(?P<kind>Table|Figure)\]\s*", re.I)
_CAPTION_KIND_RE = re.compile(
    r"^\s*(?P<kind>Table|Figure|Tbl\.?|Fig\.?)(?:\s*No\.?)?\s*",
    re.I,
)


def normalize_source_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u200b", "").replace("\u2060", "").replace("\xad", "")
    value = value.replace("．", ".").replace("。", ".").replace("‧", "·")
    # OCR engines occasionally use a middle dot for a hierarchy dot. Restrict
    # the repair to digit/digit positions so prose bullets remain untouched.
    value = re.sub(r"(?<=\d)[·∙](?=\d)", ".", value)
    return value


def _canonical_caption_kind(value: str) -> str:
    return "table" if value.casefold().startswith("t") else "figure"


@dataclass(frozen=True, slots=True)
class ParsedIdentifier:
    raw: str
    core: str
    suffixes: tuple[str, ...]
    match_key: str
    end: int

    @property
    def depth(self) -> int:
        parts = self.core.split(".")
        if len(parts) == 2 and parts[1] == "0":
            return 1
        return len(parts)


@dataclass(frozen=True, slots=True)
class ParsedCaption:
    kind: str
    identifier: ParsedIdentifier
    title: str
    raw: str


def parse_identifier_prefix(text: str) -> ParsedIdentifier | None:
    normalized = normalize_source_text(text)
    match = _PREFIX_RE.match(normalized)
    if not match:
        return None
    raw = match.group("id").strip()
    core_match = _CORE_RE.match(raw)
    if not core_match:
        return None
    core = re.sub(r"\s+", "", core_match.group(0))
    suffix_text = raw[core_match.end() :]
    suffixes: list[str] = []
    for item in _SUFFIX_RE.finditer(suffix_text):
        token = (
            item.group("paren")
            or item.group("bracket")
            or item.group("separated")
            or item.group("bare")
        )
        suffixes.append(token.casefold())
    match_key = core.casefold()
    if suffixes:
        match_key += "|" + "|".join(suffixes)
    return ParsedIdentifier(
        raw=raw,
        core=core,
        suffixes=tuple(suffixes),
        match_key=match_key,
        end=match.end(),
    )


def parse_catalog_entry(text: str) -> ParsedCaption | None:
    normalized = normalize_source_text(text)
    identifier = parse_identifier_prefix(normalized)
    if identifier is None:
        return None
    tail = normalized[identifier.end :]
    kind_match = _CATALOG_KIND_RE.match(tail)
    if not kind_match:
        return None
    return ParsedCaption(
        kind=kind_match.group("kind").lower(),
        identifier=identifier,
        title=tail[kind_match.end() :].strip(),
        raw=text,
    )


def parse_caption(text: str, expected_kind: str | None = None) -> ParsedCaption | None:
    normalized = normalize_source_text(text)
    kind_match = _CAPTION_KIND_RE.match(normalized)
    if not kind_match:
        return None
    kind = _canonical_caption_kind(kind_match.group("kind"))
    if expected_kind is not None and kind != expected_kind.casefold():
        return None
    tail = normalized[kind_match.end() :]
    identifier = parse_identifier_prefix(tail)
    if identifier is None:
        return None
    return ParsedCaption(
        kind=kind,
        identifier=identifier,
        title=tail[identifier.end :].strip(" \t:–—-"),
        raw=text,
    )


def identifier_match_key(value: str) -> str:
    parsed = parse_identifier_prefix(value)
    return parsed.match_key if parsed else normalize_source_text(value).strip().casefold()
