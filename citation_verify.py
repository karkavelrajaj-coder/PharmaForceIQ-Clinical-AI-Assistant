"""Verify [Author et al] citations against the actually retrieved papers.

Catches cross-domain citation bleed where the LLM cites a real author whose
paper was not in the current retrieval (e.g. [Grundy et al] showing up on an
NSCLC answer because the paper isn't in oncology retrieval).

Behaviour:
  - Extract every "[Surname et al]" or "[Surname et al, 2019]" token from answer.
  - Build the set of valid surnames from papers_cited[*].authors.
  - Any citation whose surname doesn't fuzzy-match a valid surname is flagged
    with a small inline "⚠️ source not retrieved" badge.

Surname matching is case-insensitive and tolerates trailing punctuation /
suffixes. Does NOT try to verify the CONTENT — only that the author is
actually one of the retrieved sources.
"""
import re
import unicodedata

# Patterns for citation tokens:
#   [Surname et al]
#   [Surname et al, 2019]
#   [Surname et al.]
#   [Surname JM et al]
#   [Surname JM et al, 2019]
#   [Smith and Jones]
_CITATION_RE = re.compile(
    r"\[\s*([A-Z][\w\-']{1,40}(?:\s+[A-Z]{1,4})?)\s+(?:et\s+al\.?|and\s+\w+)"
    r"(?:\s*,\s*\d{4})?\s*\]",
    re.IGNORECASE,
)


def _norm_surname(s: str) -> str:
    """Normalize for matching: NFKD, lowercase, strip punctuation."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z\-']", "", s)
    return s.lower()


def collect_valid_surnames(papers_cited) -> set:
    """Build the set of valid surnames from a papers_cited list.

    Each paper has an authors list. We treat the LAST whitespace-separated
    token of each author name as the surname (e.g. 'Christian Manegold' -> 'Manegold').
    """
    surnames = set()
    for p in (papers_cited or []):
        authors = p.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        for a in authors:
            if not a:
                continue
            parts = re.split(r"\s+", str(a).strip())
            # Add every non-initial token as a candidate surname so both
            # "Takashi" and "Eguchi" map (caller passes whatever surname appears in answer).
            for tok in parts:
                if len(tok) <= 2 and tok.isupper():
                    continue
                n = _norm_surname(tok)
                if len(n) >= 3:
                    surnames.add(n)
    return surnames


def extract_citations(text: str) -> list:
    """Return list of {surname, matched, span} from text."""
    if not text:
        return []
    out = []
    for m in _CITATION_RE.finditer(text):
        raw = m.group(1)
        # If the captured token has spaces ('Smith JM'), strip initials
        first_word = raw.split()[0]
        out.append({
            "surname": _norm_surname(first_word),
            "matched": m.group(0),
            "span": m.span(),
        })
    return out


def verify_citations(answer_md: str, papers_cited: list) -> dict:
    """Return stats: total / verified / unverified / unverified_items."""
    valid = collect_valid_surnames(papers_cited)
    cites = extract_citations(answer_md)
    unverified = []
    for c in cites:
        # Try exact and a couple of substring tolerances
        sn = c["surname"]
        if not sn:
            continue
        if sn in valid:
            continue
        # Tolerance: any valid surname that STARTS with the cited surname or vice versa
        # (e.g. 'Smith' would match 'Smithfield' — but only if both >= 5 chars)
        match = False
        if len(sn) >= 5:
            for v in valid:
                if len(v) >= 5 and (v.startswith(sn) or sn.startswith(v)):
                    match = True
                    break
        if not match:
            unverified.append(c)
    return {
        "total": len(cites),
        "verified": len(cites) - len(unverified),
        "unverified": len(unverified),
        "unverified_items": unverified,
        "valid_surnames": sorted(valid),
    }


def annotate_unverified_citations(answer_md: str, papers_cited: list,
                                   max_badges: int = 30) -> tuple:
    """Inline-add a small ⚠️ badge after each unverified citation token.

    Returns (annotated_markdown, stats). Right-to-left insertion to keep spans
    valid. Idempotent over re-runs (skips if a 'source not retrieved' badge
    is already adjacent).
    """
    stats = verify_citations(answer_md, papers_cited)
    if not stats["unverified_items"]:
        return answer_md, stats

    items = sorted(stats["unverified_items"], key=lambda x: x["span"][1], reverse=True)
    badge = (
        ' <span title="Citation surname not found in retrieved papers — likely from training memory"'
        ' style="color:#a82020;background:#ffe6e6;padding:0 4px;border-radius:3px;'
        'font-size:0.78rem;border:1px solid #f5b5b5;">⚠️ source not retrieved</span>'
    )
    out = answer_md
    count = 0
    seen_spans = set()
    for it in items:
        if count >= max_badges:
            break
        if it["span"] in seen_spans:
            continue
        seen_spans.add(it["span"])
        end = it["span"][1]
        nearby = out[end:end + 80]
        if "source not retrieved" in nearby:
            continue
        out = out[:end] + badge + out[end:]
        count += 1
    stats["badges_added"] = count
    return out, stats
