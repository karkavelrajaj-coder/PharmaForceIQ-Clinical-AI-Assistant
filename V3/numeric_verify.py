"""Layer 2 of Concern 2 — post-generation numeric verification.

Extracts quantitative claims from an LLM answer and checks each against the
retrieved corpus (paper_context + claims + fda). Numbers NOT found in the
corpus get a `⚠️ unverified` badge inline so the reader knows the number
was not directly quoted from any retrieved passage.

This is mechanical defense in depth — it doesn't depend on the LLM following
prompt rules, so it catches hallucinations the prompt-based rules miss.
"""
import re
import unicodedata


# Patterns of quantitative claims worth verifying.
# Each pattern captures the number itself in group 1 (and optionally group 2 for ranges).
_CLINICAL_NUMBER_PATTERNS = [
    # Percentages: "22%", "20-25%", "0.5 %"
    (re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*%"), "percent"),
    # Decimal ratio with prefix: RR 0.78, HR 0.93, OR 1.5, MD = 0.54, aOR 0.7
    (re.compile(r"\b(?:RR|HR|OR|MD|aOR|aHR|IRR|SMD)\s*=?\s*(\d+(?:\.\d+)?)\b", re.I), "ratio"),
    # CI ranges: "0.70-0.86", "1.05 to 1.34"
    (re.compile(r"\b(\d+\.\d{2,3})\s*(?:[-–—]|to)\s*(\d+\.\d{2,3})\b"), "range"),
    # LDL-C / cholesterol units mg/dL: "70 mg/dL", "<55 mg/dL", "38.7 mg/dL"
    (re.compile(r"(?:[<>]\s*)?(\d+(?:\.\d+)?)\s*mg\s*/\s*d[lL]"), "mgdl"),
    # mmol/L: "1.0 mmol/L"
    (re.compile(r"(\d+(?:\.\d+)?)\s*mmol\s*/\s*[lL]"), "mmoll"),
    # Sample sizes: N=26,497 or n=12,526
    (re.compile(r"\b[nN]\s*=\s*([\d,]+)\b"), "sample_size"),
    # p-values: p<0.05, p=0.001
    (re.compile(r"\bp\s*[<>=≤≥]\s*(\d+(?:\.\d+)?)\b", re.I), "pvalue"),
]


# Numbers we should not flag (LLM may produce these from training and they're not "claims"):
#   - bare 1-digit cardinal numbers ("1", "2", "3" used as bullet/section markers)
#   - common years 2000-2030
#   - common ages
_SKIP_NUMBER_RE = re.compile(r"^(?:[0-9]|20\d{2}|19\d{2})$")


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip diacritics, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Remove commas in numbers: "26,497" -> "26497"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text


def _number_in_corpus(num_str: str, corpus_norm: str) -> bool:
    """Check if a number appears in the normalized corpus.

    Tries a few representation variants:
      - exact: "0.78"
      - without leading zero: ".78"
      - with leading zero: "0.78" (already)
      - integer fragment for decimals: "78" (last resort, only for short decimals)
    """
    if not num_str:
        return True
    # Strip trailing punctuation
    num_str = num_str.rstrip(".,;:")
    n = num_str.replace(",", "")
    if not n:
        return True
    # Try the exact string in the corpus (normalized)
    if n in corpus_norm:
        return True
    # No-leading-zero variant: 0.78 -> .78
    if "." in n and n.startswith("0."):
        if n[1:] in corpus_norm:
            return True
    # With leading zero: .78 -> 0.78 (rare in LLM output but try)
    if n.startswith("."):
        if ("0" + n) in corpus_norm:
            return True
    # For decimals like "0.78", try trailing zero variant "0.780"
    if "." in n:
        if (n + "0") in corpus_norm:
            return True
        # And drop a trailing zero: "0.780" -> "0.78"
        if n.endswith("0") and n[:-1] in corpus_norm:
            return True
    return False


def extract_clinical_numbers(text: str) -> list:
    """Return a list of (matched_string, value_string, kind, span) tuples.

    Iterates all the patterns and collects each match. Skips section markers
    and year-looking numbers.
    """
    out = []
    if not text:
        return out
    for pat, kind in _CLINICAL_NUMBER_PATTERNS:
        for m in pat.finditer(text):
            v = m.group(1)
            if _SKIP_NUMBER_RE.match(v):
                continue
            out.append({
                "matched": m.group(0),
                "value": v,
                "kind": kind,
                "span": m.span(),
            })
            if kind == "range" and m.lastindex and m.lastindex >= 2:
                out.append({
                    "matched": m.group(0),
                    "value": m.group(2),
                    "kind": "range_high",
                    "span": m.span(),
                })
    return out


def verify_numbers(answer_md: str, corpus_texts) -> dict:
    """Verify numbers in answer_md against the given corpus texts.

    Args:
      answer_md: the LLM-generated markdown answer
      corpus_texts: iterable of strings — paper_context, claims_context, fda_context, etc.

    Returns dict:
      {
        "total": int,                 # total quantitative claims found
        "verified": int,              # found in corpus
        "unverified": int,            # NOT found in corpus
        "unverified_items": [         # list of {matched, value, kind, span}
          ...
        ],
      }
    """
    corpus = "\n".join(t or "" for t in corpus_texts)
    corpus_norm = _normalize(corpus)
    items = extract_clinical_numbers(answer_md)
    unverified = []
    for item in items:
        if not _number_in_corpus(item["value"], corpus_norm):
            unverified.append(item)
    return {
        "total": len(items),
        "verified": len(items) - len(unverified),
        "unverified": len(unverified),
        "unverified_items": unverified,
    }


def annotate_unverified(answer_md: str, corpus_texts, max_badges: int = 20) -> tuple:
    """Add a small ⚠️ badge after each numeric claim NOT found in the corpus.

    Returns (annotated_markdown, stats_dict). The annotation is HTML so it
    survives streamlit's markdown renderer (unsafe_allow_html is on).

    To avoid annotating the same number twice, we sort unverified items by
    position and process from right to left so spans stay valid.
    """
    stats = verify_numbers(answer_md, corpus_texts)
    if not stats["unverified_items"]:
        return answer_md, stats

    # De-duplicate by span; sort right-to-left so insertions don't shift earlier spans
    spans_seen = set()
    items = []
    for it in stats["unverified_items"]:
        if it["span"] in spans_seen:
            continue
        spans_seen.add(it["span"])
        items.append(it)
    items.sort(key=lambda x: x["span"][1], reverse=True)

    badge = (
        ' <span title="Number not found in retrieved evidence — verify against source"'
        ' style="color:#b85c00;background:#fff3e0;padding:0 4px;border-radius:3px;'
        'font-size:0.78rem;border:1px solid #ffd9a8;">⚠️ unverified</span>'
    )
    out = answer_md
    count = 0
    for it in items:
        if count >= max_badges:
            break
        end = it["span"][1]
        # Don't double-badge if already badged (idempotency for streamlit reruns)
        nearby = out[end:end + 60]
        if "unverified" in nearby:
            continue
        out = out[:end] + badge + out[end:]
        count += 1
    stats["badges_added"] = count
    return out, stats
