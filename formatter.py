"""
Clinical Knowledge Context Formatter
======================================
Formats retrieved Qdrant passages (from clinical_guidelines, papers, or trials collections)
into a structured text block for injection into the synthesis LLM prompt.

Abstract passages emit a pre-built CITE badge and a DOI-free reference line so
the LLM can copy the badge directly without parsing or constructing hyperlinks:

    [N] CITE: [[Author et al., Year]](https://doi.org/DOI)
        REF:  Full author list. (Year). Title. Journal.
        passage text ...

    The DOI is intentionally omitted from REF so the LLM never sees a raw DOI
    string to copy.  The only citable link is the pre-built CITE badge.

    When no DOI is available the badge degrades to bold text:
        [N] CITE: **[Author et al., Year]**
            REF:  ...

Guideline passages use topic/section labels (source name intentionally omitted):
    [N] Therapeutic Area — Category | Section | Page P
        passage text ...

The References section appended to the final response is built separately by
_build_references_block() in chat.py — see retriever.py module docstring for
the format details.
"""

from __future__ import annotations


def _short_author_label(authors: str) -> str:
    """
    Convert a full author string into a short 'First et al.' label.

    Handles common formats:
      "Smith, John; Jones, Mary"  → "Smith et al."
      "John Smith; Mary Jones"    → "Smith et al."
      "Smith JA"                  → "Smith"
    """
    if not authors:
        return "Author unknown"
    # Split on semicolons (common export format) first, then commas
    for sep in (";", ","):
        parts = [p.strip() for p in authors.split(sep) if p.strip()]
        if len(parts) >= 2:
            first = parts[0]
            # "Last, First" → take token before first comma
            last_name = first.split(",")[0].strip()
            return f"{last_name} et al."
    # Single author
    return authors.split(",")[0].split(";")[0].strip()


def _normalise_doi(doi: str) -> str:
    """Strip any URL or label prefix from a DOI string, returning the bare DOI."""
    doi = doi.strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "DOI: ",
        "DOI:",
        "doi: ",
        "doi:",
    ):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):].strip()
            break
    return doi


def build_abstract_badge(p: dict) -> str | None:
    """
    Return the pre-formatted Markdown badge for an abstract passage, or None
    if insufficient metadata is available.

    Format when DOI present:
        [[Author et al., Year]](https://doi.org/DOI)

    Format when no DOI:
        **[Author et al., Year]**
    """
    authors = (p.get("authors") or "").strip()
    year    = str(p.get("year") or "").strip()
    doi     = (p.get("doi")     or "").strip()

    if not (authors or year):
        return None

    label = _short_author_label(authors) if authors else "Unknown"
    if year:
        label = f"{label}, {year}"

    if doi:
        doi_clean = _normalise_doi(doi)
        return f"[[{label}]](https://doi.org/{doi_clean})"
    else:
        return f"**[{label}]**"


def build_abstract_citation(p: dict) -> str:
    """
    Build a formatted bibliographic citation string from passage metadata.

    Produces: Authors (Year). Title. Journal. DOI: ...
    Falls back gracefully when individual fields are absent.
    """
    authors = (p.get("authors") or "").strip()
    year    = (p.get("year")    or "").strip()
    title   = (p.get("title")   or "").strip()
    journal = (p.get("journal") or "").strip()
    doi     = (p.get("doi")     or "").strip()

    parts: list[str] = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year}).")
    elif parts:
        parts[-1] += "."
    if title:
        parts.append(f"{title}.")
    if journal:
        parts.append(f"{journal}.")
    if doi:
        parts.append(f"DOI: {doi}")

    return " ".join(parts) if parts else (title or "Unknown reference")


def format_paper_context(passages: list[dict]) -> str:
    """
    Format a list of retrieved passages into a citation-annotated text block.

    Paper passages (db == "papers") emit a full bibliographic citation plus
    a pre-built CITE_AS badge line so the LLM can copy it directly without parsing
    DOI strings.  Guideline passages emit a topic/section label (source database
    name is intentionally omitted).

    Parameters
    ----------
    passages : list[dict]
        Each dict produced by ``retriever.retrieve_papers()``.

    Returns
    -------
    str
        A formatted string block ready for injection into an LLM prompt.
    """
    if not passages:
        return "(No relevant clinical guidelines or reference documents found.)"

    lines: list[str] = []
    lines.append("=== CLINICAL KNOWLEDGE EVIDENCE ===\n")

    for idx, p in enumerate(passages, start=1):
        db_label  = p.get("db", "")
        ta        = p.get("therapeutic_area", "")
        category  = p.get("category", "")
        section   = p.get("section", "")
        page      = p.get("page", 0)
        text      = p.get("text", "").strip()
        score     = p.get("relevance_score")
        chunk_idx = p.get("chunk_index", 0)
        total     = p.get("total_chunks", 0)

        if db_label == "papers":
            badge = build_abstract_badge(p)

            if badge:
                # Badge is the ONLY citation identifier shown — the LLM has nothing
                # else to copy. Full bibliographic detail follows on the next line for
                # author/journal context but does NOT contain a raw DOI string.
                authors = (p.get("authors") or "").strip()
                year    = str(p.get("year") or "").strip()
                title   = (p.get("title")   or "").strip()
                journal = (p.get("journal") or "").strip()

                full_ref_parts = []
                if authors:
                    full_ref_parts.append(authors)
                if year:
                    full_ref_parts.append(f"({year})")
                if title:
                    full_ref_parts.append(f"{title}.")
                if journal:
                    full_ref_parts.append(journal)
                # DOI intentionally omitted here — it is already encoded in the badge URL
                full_ref = ". ".join(full_ref_parts) if full_ref_parts else title or "Unknown"

                citation = f"[{idx}] CITE: {badge}\n    REF:  {full_ref}"
            else:
                # No DOI available — show full citation so the LLM can build **[Author, Year]**
                citation = f"[{idx}] {build_abstract_citation(p)}"

            if ta or category:
                topic = f"{ta} — {category}" if ta and category else (ta or category)
                citation += f"  [{topic}]"
        else:
            # Topic/section label for guideline chunks (source name omitted)
            if ta and category:
                topic = f"{ta} — {category}"
            elif ta:
                topic = ta
            else:
                topic = p.get("title", "")
            citation = f"[{idx}] {topic} | {section} | Page {page}"
            if total > 1:
                citation += f" (chunk {chunk_idx + 1}/{total})"

        if score is not None:
            citation += f"  (relevance: {score:.2f})"
        lines.append(citation)

        indented = "\n    ".join(text.split("\n"))
        lines.append(f"    {indented}")
        lines.append("")

    lines.append("=== END CLINICAL KNOWLEDGE EVIDENCE ===")
    return "\n".join(lines)


def format_inline_citations(passages: list[dict]) -> dict[int, str]:
    """
    Return a mapping of citation number → short citation string for use
    in inline references within the synthesis response.

    Abstract passages: "Authors (Year). Title."
    Guideline passages: "Therapeutic Area — Category | Page N"
    """
    citations: dict[int, str] = {}
    for idx, p in enumerate(passages, start=1):
        if p.get("db") == "papers":
            citations[idx] = build_abstract_citation(p)
        else:
            ta       = p.get("therapeutic_area", "")
            category = p.get("category", "")
            page     = p.get("page", 0)
            topic    = f"{ta} — {category}" if ta and category else (ta or category or "")
            citations[idx] = f"{topic} | Page {page}".strip(" |")

    return citations
