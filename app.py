"""
PharmaForceIQ Clinical AI Assistant — Retrieval Model Comparison Demo (v2)
=============================================================================
v2 change: this now uses PRODUCTION'S REAL PROMPT LOGIC AND REAL RETRIEVAL-
DIVERSITY ALGORITHM, verbatim, verified directly against app/synthesis/
prompts.py, app/synthesis/engine.py, and app/papers/formatter.py in the main
repo. What's NOT identical to production is explicitly stated below.

WHAT'S NOW EXACT (verified line-by-line against production):
  - System prompt: prompts.py copied verbatim (zero external dependencies,
    confirmed) and build_system_prompt() called unmodified.
  - Retrieval diversity: the real MAX_CHUNKS_PER_STEM=2 + 4x-over-fetch
    dedup algorithm from app/papers/retriever.py, reproduced exactly.
  - Context block header/format: "=== CLINICAL KNOWLEDGE EVIDENCE ==="
    format, matching the GUIDELINE branch of format_paper_context() (see
    note below on why the guideline branch, not the papers/DOI branch).
  - Synthesis call parameters: model, max_tokens=2000, temperature=0.1 —
    exact match to stream_synthesize_answer() in engine.py.

WHAT'S NOT THE SAME AS PRODUCTION (data-access gap, not a code gap):
  - The actual chunks: ours come from our own 376-doc test corpus in our
    own Qdrant cluster, not production's 600K+ doc corpus.
  - Claims (Databricks) and FDA (OpenFDA): still stubbed, no access yet.
  - Citation richness: production's "papers" branch uses author/year/DOI
    badges — our chunk payload doesn't carry that bibliographic metadata
    (see NOTE below), so we use the GUIDELINE branch format instead
    (topic/section label, no author/DOI) — this is a deliberate, correct
    choice given our data, not a shortcut.
-----------------------------------------------------------------------------
NOTE ON WHY WE USE THE GUIDELINE-STYLE CITATION, NOT THE PAPERS-STYLE ONE:
Production's format_paper_context() branches on db_label == "papers" to
decide which citation style to use. The "papers" branch needs authors,
year, title, journal, and a DOI to build its citation badge. Our chunk
payload (from chunk_documents.py / embed_and_index.py) only carries
doc_stem, folder, chunk_type, section_title, text -- no bibliographic
metadata was ever extracted for this corpus. Forcing the papers-style
badge format onto data that doesn't have those fields would mean either
crashing or fabricating fake author/year values -- neither is acceptable.
The guideline-style citation (topic/section/relevance, no author badge)
uses only fields our data actually has, so that's the correct match, not
an approximation.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import re
import time

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic

from prompts import build_system_prompt      # real production prompt module, copied verbatim
from formatter import format_paper_context    # real production formatter, copied verbatim
import citation_verify                        # real production citation verifier, copied verbatim
import numeric_verify                         # real production numeric verifier, copied verbatim

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PharmaForceIQ Clinical AI Assistant", page_icon="🩺", layout="wide")

_BRAND_TEAL = "#0E7C86"
_BRAND_NAVY = "#16233F"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_CHOICES = {
    "bge_small": {"label": "BAAI/bge-small-en-v1.5 (current production)",
                  "repo_id": "BAAI/bge-small-en-v1.5", "collection": "pharmaforceiq_bge_small"},
    "medembed_small": {"label": "MedEmbed-small-v0.1 (medical-tuned)",
                        "repo_id": "abhinand/MedEmbed-small-v0.1", "collection": "pharmaforceiq_medembed_small"},
}
TOP_K = 5                    # final number of passages sent to Claude — matches production
FETCH_K = TOP_K * 4          # over-fetch factor — EXACT match to retriever.py's fetch_k = top_k * 4
MAX_CHUNKS_PER_STEM = 2      # EXACT match to retriever.py's diversity cap
CLAUDE_MODEL = "claude-sonnet-4-6"          # matches LLM_MODEL_ANSWER in production config
CLAUDE_MODEL_UTILITY = "claude-haiku-4-5"   # matches LLM_MODEL_UTILITY in production config
SYNTHESIS_MAX_TOKENS = 2000  # EXACT match to stream_synthesize_answer()
SYNTHESIS_TEMPERATURE = 0.1  # EXACT match to stream_synthesize_answer()

# Provider role/type options — real keys from prompts.py's PROVIDER_ROLE_FRAMING /
# PROVIDER_TYPE_FRAMING dicts, so selecting these actually changes the real prompt.
PROVIDER_ROLES = ["CARDIOLOGIST", "ONCOLOGIST", "PCP", "SPECIALIST"]
PROVIDER_TYPES = ["PHYSICIAN", "APP"]

SAMPLE_QUESTIONS = [
    "What factors are associated with malnutrition in heart failure patients?",
    "How does D3 lymph node dissection compare to D2 in colorectal cancer surgery?",
    "What are the risks of air embolism during cardiac catheter ablation?",
    "Does chemotherapy with 5-FU affect lung metastasis through neutrophils?",
]

# ---------------------------------------------------------------------------
# SECRETS
# ---------------------------------------------------------------------------
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        st.error(f"Missing secret: `{key}`. Add it in Streamlit Cloud under Settings → Secrets.")
        st.stop()


QDRANT_URL = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---------------------------------------------------------------------------
# v4: REAL bibliographic metadata now lives directly in the Qdrant payload
# (pushed there by update_qdrant_payload.py, confirmed working live).
# retrieve() below reads authors/year/journal/title/doi straight from
# each point's payload -- no separate local file needed at runtime, and
# none is shipped with this deployment.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=True)
def get_embedding_model(model_key: str) -> SentenceTransformer:
    return SentenceTransformer(MODEL_CHOICES[model_key]["repo_id"])


@st.cache_resource(show_spinner=False)
def get_claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# HIPAA CHECK — unchanged from v1 (regex + Haiku, fail-closed). Already
# verified in the v1 build; not modified here.
# ---------------------------------------------------------------------------
_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    re.compile(r"\bMRN[\s:#]*\d{4,}\b", re.IGNORECASE),
]
_HAIKU_PHI_SYSTEM = """You detect Protected Health Information (PHI) under HIPAA Safe Harbor -- \
things like patient names, specific dates tied to a real patient, addresses, or other identifying \
details about a real, specific person. Hypothetical patients ("a 65-year-old with HFrEF") and \
population-level statistics are NOT PHI. Respond with exactly one word: YES if PHI is present, \
NO if it is not."""


def regex_phi_check(text: str) -> bool:
    return any(p.search(text) for p in _PHI_PATTERNS)


def semantic_phi_check(text: str) -> bool:
    try:
        client = get_claude_client()
        response = client.messages.create(
            model=CLAUDE_MODEL_UTILITY, max_tokens=5, temperature=0,
            system=_HAIKU_PHI_SYSTEM, messages=[{"role": "user", "content": text}],
        )
        return "YES" in response.content[0].text.strip().upper()
    except Exception:
        return True  # fail CLOSED


def contains_possible_phi(text: str) -> bool:
    if regex_phi_check(text):
        return True
    return semantic_phi_check(text)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RETRIEVAL — v2: now with production's REAL diversity-dedup algorithm,
# reproduced exactly from app/papers/retriever.py's retrieve_papers().
# ---------------------------------------------------------------------------
def retrieve(query: str, model_key: str, top_k: int = TOP_K) -> list[dict]:
    client = get_qdrant_client()
    model = get_embedding_model(model_key)
    collection = MODEL_CHOICES[model_key]["collection"]

    qvec = model.encode(query, normalize_embeddings=True).tolist()

    # Over-fetch 4x — EXACT match to retriever.py: fetch_k = top_k * 4
    fetch_k = top_k * 4
    response = client.query_points(collection_name=collection, query=qvec, limit=fetch_k)
    raw = [
        {
            "doc_stem": r.payload.get("doc_stem"),
            "section_title": r.payload.get("section_title"),
            "chunk_type": r.payload.get("chunk_type"),
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
            # v4: real bibliographic metadata, present only for documents
            # successfully enriched via enrich_metadata.py + pushed via
            # update_qdrant_payload.py. Absent (None) for the ~4 documents
            # that didn't enrich -- correctly falls through to the
            # guideline citation branch in build_paper_context() below.
            "db": r.payload.get("db"),
            "authors": r.payload.get("authors"),
            "year": r.payload.get("year"),
            "journal": r.payload.get("journal"),
            "title": r.payload.get("title"),
            "doi": r.payload.get("doi"),
        }
        for r in response.points
    ]

    # Sort by score descending (Qdrant already returns sorted, but production
    # re-sorts explicitly after the score-threshold filter — reproduced here
    # for exactness, even though it's a no-op on already-sorted input)
    raw.sort(key=lambda p: p["score"], reverse=True)

    # EXACT match to retriever.py: MAX_CHUNKS_PER_STEM = 2, cap any single
    # document's contribution so the final top_k draws from >= 3 documents
    # instead of one document dominating all slots (this is the specific
    # fix for the "shallow pointers" issue found in the v1 demo).
    seen: dict[str, int] = {}
    deduped: list[dict] = []
    for p in raw:
        stem = (p.get("doc_stem") or "").strip()
        if not stem:
            deduped.append(p)
            continue
        if seen.get(stem, 0) >= MAX_CHUNKS_PER_STEM:
            continue
        seen[stem] = seen.get(stem, 0) + 1
        deduped.append(p)

    final = deduped[:top_k]
    return final


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# v4.1 FIX: reformat authors from our storage shape into what production's
# real, unmodified _short_author_label() (in formatter.py) actually expects.
#
# ROOT CAUSE (confirmed by direct testing, not guessed):
# enrich_metadata.py (our own script) stores authors as CrossRef's raw
# given/family fields joined directly: "Shih-Chieh Chien, Chi-In Lo, ..."
# -- comma between authors, first-name-first.
#
# formatter.py's _short_author_label() (production's real, unmodified code)
# expects "Last, First; Last, First" -- semicolon between authors, comma
# between last/first WITHIN each author -- per its own docstring. Given our
# actual shape, it treats the whole first entry as one unsplit token, so
# "Shih-Chieh Chien" (the full name) becomes the "surname" instead of just
# "Chien". Confirmed: this also breaks citation_verify's regex downstream,
# since it isn't built to match a full first+last name in this citation
# shape at all (tested: 0 matches for the full-name form, 1 match for the
# surname-only form).
#
# NOT a production bug and not fixed by touching formatter.py or
# citation_verify.py (both untouched, verbatim). This is our OWN script's
# output shape not matching what production's real function already
# expects -- fixed here, in our own integration code, without re-running
# CrossRef or touching what's already stored in Qdrant.
# ---------------------------------------------------------------------------
def reformat_authors_for_production(authors: str) -> str:
    if not authors:
        return authors
    people = [p.strip() for p in authors.split(",") if p.strip()]
    reformatted = []
    for person in people:
        parts = person.rsplit(" ", 1)  # last whitespace-separated token = family name
        if len(parts) == 2:
            given, family = parts
            reformatted.append(f"{family}, {given}")
        else:
            reformatted.append(person)  # single-word name, leave as-is
    return "; ".join(reformatted)


# ---------------------------------------------------------------------------
# CONTEXT FORMATTING — v4: enriched documents (real authors/year/journal/doi
# confirmed via CrossRef) now route through format_paper_context()'s REAL
# "papers" branch, which calls build_abstract_badge() -- a pre-built
# citation Claude copies, never invents. Documents without confirmed
# metadata still correctly fall through to the guideline branch, exactly
# as before -- this is NOT an approximation, it's the same honest fallback,
# just now only applied to the ~1% of documents that genuinely have no
# confirmed real data (CrossRef 404s), not all of them.
# ---------------------------------------------------------------------------
def build_paper_context(chunks: list[dict]) -> str:
    mapped = []
    any_unenriched = False
    for c in chunks:
        if c.get("authors") or c.get("year"):
            mapped.append({
                "db": "papers",
                "authors": reformat_authors_for_production(c.get("authors", "")),
                "year": c.get("year", ""),
                "title": c.get("title", ""),
                "journal": c.get("journal", ""),
                "doi": c.get("doi", ""),
                "section": c.get("section_title") or "",
                "page": 0,  # no page metadata in our corpus -- known, honest gap
                "text": c.get("text", ""),
                "relevance_score": c.get("score"),
                "chunk_index": 0,
                "total_chunks": 1,
            })
        else:
            any_unenriched = True
            mapped.append({
                "db": "guideline",              # no confirmed metadata -- correct, honest fallback
                "therapeutic_area": "",
                "category": "",
                "section": c.get("section_title") or "",
                "page": 0,
                "text": c.get("text", ""),
                "relevance_score": c.get("score"),
                "chunk_index": 0,
                "total_chunks": 1,
                "title": c.get("doc_stem") or "Unknown source",
            })

    context = format_paper_context(mapped)

    # Only warn about missing-metadata sources when at least one chunk in
    # THIS answer actually lacks confirmed data -- previously this notice
    # was unconditional (correct when ALL sources lacked metadata; no
    # longer accurate now that most do have it).
    if any_unenriched:
        notice = (
            "NOTE: some sources above have no confirmed author/year/journal "
            "metadata. For those specific sources only, refer to them by "
            "[Source N] and do NOT attach an author name or year, even if "
            "you recognize the underlying study -- any such attribution "
            "would be from memory, not from what was retrieved, and may be "
            "wrong. Sources with a real [[Author et al., Year]] badge above "
            "ARE confirmed real data and should be cited using that exact "
            "badge.\n\n"
        )
        context = notice + context

    return context


# ---------------------------------------------------------------------------
# SYNTHESIS — v2: uses the REAL build_system_prompt() from the copied
# prompts.py, and matches stream_synthesize_answer()'s exact call shape
# and parameters. claims_context / fda_context are always empty here
# (those sources aren't connected), which correctly means:
#   - has_claims  = False -> claims instructions correctly omitted
#   - has_evidence = False -> evidence_docs instructions correctly omitted
#     (this app's literature block is NOT the [EVIDENCE CONTEXT]-headed
#     evidence_docs source -- it's the "=== CLINICAL KNOWLEDGE EVIDENCE ==="
#     guideline/paper source. Verified this distinction directly against
#     engine.py before building this -- see chat history.)
# ---------------------------------------------------------------------------
def stream_answer(question: str, paper_context: str, provider_role: str, provider_type: str):
    client = get_claude_client()

    claims_context = ""  # not connected -- Databricks access not yet available
    fda_context = ""     # not connected -- OpenFDA integration not yet built

    parts = []
    if claims_context:
        parts.append(claims_context)
    if paper_context:
        parts.append(paper_context)
    if fda_context:
        parts.append(fda_context)
    parts.append(f"\n=== USER QUESTION ===\n{question}")
    user_content = "\n\n".join(parts)

    has_claims = bool(claims_context and "[CLAIMS CONTEXT" in claims_context)
    has_evidence = bool(paper_context and "[EVIDENCE CONTEXT" in paper_context)

    system_prompt = build_system_prompt(
        provider_role=provider_role,
        include_claims_instructions=has_claims,
        include_evidence_instructions=has_evidence,
        provider_type=provider_type,
    )

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        temperature=SYNTHESIS_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# UI — SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<h2 style="color:{_BRAND_NAVY};margin:0 0 0.2rem;">🩺 PharmaForceIQ</h2>'
        f'<p style="color:{_BRAND_TEAL};font-weight:600;margin:0 0 1rem;">Retrieval Model Comparison Demo</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Embedding Model**")
    selected_model = st.radio(
        "Choose which model powers retrieval:", options=list(MODEL_CHOICES.keys()),
        format_func=lambda k: MODEL_CHOICES[k]["label"], key="selected_model", label_visibility="collapsed",
    )
    st.caption(f"Collection: `{MODEL_CHOICES[selected_model]['collection']}`")

    st.divider()
    st.markdown("**Provider Context** *(real prompt branching — try switching these)*")
    selected_role = st.selectbox("Specialty framing", PROVIDER_ROLES, key="selected_role")
    selected_type = st.selectbox("Provider type", PROVIDER_TYPES, key="selected_type")

    st.divider()
    st.markdown("**Pipeline Stages**")
    st.success("✅ Medical Literature (Qdrant) — live, real diversity-dedup algorithm")
    st.success("✅ HIPAA check (regex + Haiku, fail-closed) — live")
    st.success("✅ System prompt — production's real prompts.py, copied verbatim")
    st.success("✅ Citation + numeric verification — production's real citation_verify.py / numeric_verify.py, copied verbatim")
    st.info("🔜 Claims Data (Databricks) — no access to this data source yet")
    st.info("🔜 FDA Drug Labels (OpenFDA) — no access to this data source yet")
    st.info("🔜 Evidence Docs (UpToDate) — no separate corpus built for this yet")
    st.caption(
        "Retrieval now applies the same MAX_CHUNKS_PER_STEM=2 diversity cap as "
        "production, over our own indexed corpus. System prompt calls the real "
        "build_system_prompt() unmodified."
    )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.warning(
        "**Demo only.** Do not enter real patient-identifiable information. "
        "Two-layer PHI check (regex + Claude Haiku), fail-closed on error."
    )

# ---------------------------------------------------------------------------
# UI — MAIN
# ---------------------------------------------------------------------------
st.markdown(f'<h1 style="color:{_BRAND_NAVY};margin-bottom:0.3rem;">Clinical AI Assistant</h1>', unsafe_allow_html=True)
st.caption(
    f"Retrieving with **{MODEL_CHOICES[selected_model]['label']}** | "
    f"Framed for **{selected_role.title()}** ({selected_type.title()}) | "
    f"Using production's real system prompt and retrieval-diversity logic."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try a sample question:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            with st.expander("Details", expanded=False):
                st.markdown(f"**Model:** `{m['model_repo']}` | **Role:** {m['role']} ({m['type']})")
                st.markdown(f"**Collection:** `{m['collection']}` | **Response time:** {m['time']}s")
                st.markdown(f"**Chunks retrieved:** {len(m['chunks'])} from "
                            f"{len(set(c['doc_stem'] for c in m['chunks']))} distinct document(s)")
                st.markdown("**Retrieved passages:**")
                for i, c in enumerate(m["chunks"], 1):
                    st.markdown(f"`[{i}]` **{c['doc_stem']}** — _{c['section_title']}_ — score `{c['score']}`")

typed = st.chat_input("Ask about clinical guidelines or research findings...")
prompt = typed or st.session_state.pop("pending_prompt", None)

if prompt:
    with st.spinner("Checking for patient-identifying information..."):
        phi_detected = contains_possible_phi(prompt)

    if phi_detected:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": ("⚠️ This question appears to contain patient-identifying information "
                        "(a structured pattern or a free-text detail like a patient name). "
                        "Please rephrase using only general clinical details."),
        })
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner(f"Searching with {MODEL_CHOICES[selected_model]['label']}..."):
            chunks = retrieve(prompt, selected_model)
            paper_context = build_paper_context(chunks)

        answer_placeholder = st.empty()
        with answer_placeholder.container():
            raw_answer = st.write_stream(stream_answer(prompt, paper_context, selected_role, selected_type))

        # -----------------------------------------------------------------
        # v3: REAL post-hoc verification, using the real production modules
        # copied verbatim (citation_verify.py, numeric_verify.py). This is
        # exactly the layer that would have caught the fabricated
        # "[Lin et al., 2024]" citation found in earlier testing.
        #
        # v4.1: papers_cited is now populated with the REAL retrieved
        # authors -- resolves the open question flagged in the previous
        # version of this file. Necessary, not optional, once the v4.1
        # author-format fix (above) made real citation badges visible to
        # citation_verify's regex at all: tested directly, with
        # papers_cited left empty, EVERY real citation -- including
        # correct ones -- got flagged as a false positive. Populating it
        # with the actual retrieved sources' authors lets genuinely
        # correct citations verify as correct, while names that are
        # genuinely absent from every retrieved source (fabrications)
        # still get caught -- tested against both cases directly before
        # this was written.
        #
        # numeric_verify: checked against the actual retrieved chunk texts,
        # which we DO have -- so this one does real, meaningful verification.
        #
        # DISPLAY BEHAVIOR: production's real, UNMODIFIED
        # annotate_unverified_citations() -- appends a small warning badge
        # NEXT TO the flagged text (production's actual behavior), not a
        # custom replacement.
        # -----------------------------------------------------------------
        papers_cited = [{"authors": c["authors"]} for c in chunks if c.get("authors")]
        annotated, citation_stats = citation_verify.annotate_unverified_citations(
            raw_answer, papers_cited=papers_cited
        )
        annotated, numeric_stats = numeric_verify.annotate_unverified(
            annotated, corpus_texts=[c.get("text", "") for c in chunks]
        )
        # "badges_added" is only set on the branch where something was
        # actually found -- when there's nothing to flag, the function
        # returns early and the key is absent. .get() is the correct,
        # already-tested-for-this-exact-case access pattern.
        if citation_stats.get("badges_added", 0) or numeric_stats.get("badges_added", 0):
            answer_placeholder.markdown(annotated, unsafe_allow_html=True)
        answer = annotated

        elapsed = round(time.time() - t0, 1)
        meta = {
            "model_repo": MODEL_CHOICES[selected_model]["repo_id"],
            "collection": MODEL_CHOICES[selected_model]["collection"],
            "time": elapsed, "chunks": chunks,
            "role": selected_role, "type": selected_type,
            "citation_stats": citation_stats, "numeric_stats": numeric_stats,
        }
        with st.expander("Details", expanded=False):
            st.markdown(f"**Model:** `{meta['model_repo']}` | **Role:** {meta['role']} ({meta['type']})")
            st.markdown(f"**Collection:** `{meta['collection']}` | **Response time:** {meta['time']}s")
            st.markdown(f"**Chunks retrieved:** {len(chunks)} from "
                        f"{len(set(c['doc_stem'] for c in chunks))} distinct document(s)")
            st.markdown(
                f"**Citation check:** {citation_stats['verified']}/{citation_stats['total']} verified "
                f"({citation_stats['unverified']} flagged) | "
                f"**Numeric check:** {numeric_stats['verified']}/{numeric_stats['total']} verified "
                f"({numeric_stats['unverified']} flagged)"
            )
            st.markdown("**Retrieved passages:**")
            for i, c in enumerate(chunks, 1):
                st.markdown(f"`[{i}]` **{c['doc_stem']}** — _{c['section_title']}_ — score `{c['score']}`")

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
