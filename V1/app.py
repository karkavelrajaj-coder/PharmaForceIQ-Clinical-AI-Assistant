"""
PharmaForceIQ Clinical AI Assistant — Retrieval Model Comparison Demo
========================================================================
A standalone Streamlit app, structured to mirror the production chat.py
pipeline (auth-less demo version), but with only ONE stage actually wired
to real data right now: Medical Literature retrieval via Qdrant.

TOGGLE: switch between BAAI/bge-small-en-v1.5 (current production model)
and abhinand/MedEmbed-small-v0.1 (medical-tuned) mid-conversation to
compare retrieval + answer quality side by side, live, in front of anyone
looking at the deployed demo.

The other production pipeline stages (Claims/Databricks, FDA/OpenFDA,
Evidence Docs, full HIPAA+Haiku check, R1-R24 claims rules) are
intentionally represented as visible "Coming Soon" stubs in the sidebar,
not hidden — so the architecture stays legible and each stage can be
wired in later without restructuring the app. See the "PIPELINE STAGES"
section below for exactly where each one plugs in.

------------------------------------------------------------------------------
DEPLOYMENT — Streamlit Community Cloud (free tier)
------------------------------------------------------------------------------
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io -> "New app" -> point at this repo,
   branch, and set the main file path to: app.py
3. In the app's "Settings -> Secrets" (NOT in this repo, NOT in git),
   paste the contents of .streamlit/secrets.toml.example with your real
   values filled in. Never commit real secrets to GitHub.
4. Deploy. First load will be slower (~30-60s) while both embedding
   models download and cache; subsequent loads are fast.

Local development: copy .streamlit/secrets.toml.example to
.streamlit/secrets.toml, fill in real values, and that file is already
gitignored.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import re
import time
from datetime import datetime

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaForceIQ Clinical AI Assistant",
    page_icon="🩺",
    layout="wide",
)

_BRAND_TEAL = "#0E7C86"
_BRAND_NAVY = "#16233F"

# ---------------------------------------------------------------------------
# CONFIG — model/collection pairs. This IS the toggle.
# ---------------------------------------------------------------------------
MODEL_CHOICES = {
    "bge_small": {
        "label": "BAAI/bge-small-en-v1.5 (current production)",
        "repo_id": "BAAI/bge-small-en-v1.5",
        "collection": "pharmaforceiq_bge_small",
    },
    "medembed_small": {
        "label": "MedEmbed-small-v0.1 (medical-tuned)",
        "repo_id": "abhinand/MedEmbed-small-v0.1",
        "collection": "pharmaforceiq_medembed_small",
    },
}
TOP_K = 5
CLAUDE_MODEL = "claude-sonnet-4-6"

SAMPLE_QUESTIONS = [
    "What factors are associated with malnutrition in heart failure patients?",
    "How does D3 lymph node dissection compare to D2 in colorectal cancer surgery?",
    "What are the risks of air embolism during cardiac catheter ablation?",
    "Does chemotherapy with 5-FU affect lung metastasis through neutrophils?",
]

# ---------------------------------------------------------------------------
# SECRETS — read from Streamlit's secrets manager only. Never hardcoded.
# ---------------------------------------------------------------------------
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        st.error(
            f"Missing secret: `{key}`. Add it in Streamlit Cloud under "
            f"**Settings → Secrets** (see `.streamlit/secrets.toml.example` "
            f"in this repo for the exact format)."
        )
        st.stop()


QDRANT_URL = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# CACHED RESOURCES — loaded once per running app instance, not per user
# session, so switching the toggle doesn't reload a model that's already
# warm. st.cache_resource is the correct primitive for this (unlike
# st.cache_data, which is for serializable return values).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@st.cache_resource(show_spinner=True)
def get_embedding_model(model_key: str) -> SentenceTransformer:
    return SentenceTransformer(MODEL_CHOICES[model_key]["repo_id"])


@st.cache_resource(show_spinner=False)
def get_claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# HIPAA CHECK — two layers, matching production's design in app/hipaa.py:
#   Layer 1: regex (deterministic, catches structured patterns instantly)
#   Layer 2: Claude Haiku (semantic, catches free-text PHI regex can't --
#            e.g. "my patient John Martinez, seen last Tuesday, has...")
# Layer 2 is genuinely addable here without new credentials, since the
# Anthropic key is already configured for synthesis -- unlike Claims/FDA,
# which need Databricks/OpenFDA access this demo doesn't have.
#
# ONE DELIBERATE DIFFERENCE FROM PRODUCTION: this fails CLOSED (blocks) on
# any Haiku API error, not open. Production's fail-open behavior was
# flagged as the #1 priority finding in the original architecture review
# -- fixing it here rather than reproducing it.
# ---------------------------------------------------------------------------
_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),          # phone
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),          # email
    re.compile(r"\bMRN[\s:#]*\d{4,}\b", re.IGNORECASE),        # MRN
]

_HAIKU_PHI_SYSTEM = """You detect Protected Health Information (PHI) under HIPAA Safe Harbor -- \
things like patient names, specific dates tied to a real patient, addresses, or other identifying \
details about a real, specific person. Hypothetical patients ("a 65-year-old with HFrEF") and \
population-level statistics are NOT PHI. Respond with exactly one word: YES if PHI is present, \
NO if it is not."""


def regex_phi_check(text: str) -> bool:
    return any(p.search(text) for p in _PHI_PATTERNS)


def semantic_phi_check(text: str) -> bool:
    """Layer 2. Returns True if PHI is detected OR if the check itself fails
    (fail closed -- block rather than silently let a question through)."""
    try:
        client = get_claude_client()
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            temperature=0,
            system=_HAIKU_PHI_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        answer = response.content[0].text.strip().upper()
        return "YES" in answer
    except Exception:
        return True  # fail CLOSED -- the fix for production's fail-open bug


def contains_possible_phi(text: str) -> bool:
    """Two-layer check. Regex first (instant, catches structured patterns);
    only calls Haiku if regex passes clean, to avoid paying for an API call
    when the answer is already known."""
    if regex_phi_check(text):
        return True
    return semantic_phi_check(text)


# ---------------------------------------------------------------------------
# RETRIEVAL — the one stage that's actually live right now
# ---------------------------------------------------------------------------
def retrieve(query: str, model_key: str, top_k: int = TOP_K) -> list[dict]:
    client = get_qdrant_client()
    model = get_embedding_model(model_key)
    collection = MODEL_CHOICES[model_key]["collection"]

    qvec = model.encode(query, normalize_embeddings=True).tolist()
    response = client.query_points(collection_name=collection, query=qvec, limit=top_k)

    return [
        {
            "doc_stem": r.payload.get("doc_stem"),
            "section_title": r.payload.get("section_title"),
            "chunk_type": r.payload.get("chunk_type"),
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
        }
        for r in response.points
    ]


# ---------------------------------------------------------------------------
# SYNTHESIS — simplified system prompt. Production has an 11-role,
# 931-line prompt with R1-R24 claims-data rules (see synthesis/prompts.py
# in the main repo) -- most of those rules are specific to interpreting
# insurance claims data, which isn't connected in this demo at all, so
# they're not meaningfully applicable here yet. This is a deliberately
# smaller prompt scoped to what's actually running: literature retrieval.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical evidence assistant. Answer the provider's question using ONLY the retrieved passages provided below. Follow these rules strictly:

1. Ground every claim in the retrieved passages. Do not use outside knowledge to fill gaps.
2. If the retrieved passages don't fully answer the question, say so explicitly rather than guessing.
3. Cite sources inline using [Source N] referring to the numbered passages below.
4. Keep the answer clinically precise and concise.
5. This is a retrieval-quality demo, not a validated clinical decision support tool. Do not present the answer as a treatment recommendation.

Retrieved passages:
{context}
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] (doc: {c['doc_stem']}, section: {c['section_title']}, "
            f"relevance: {c['score']})\n{c['text']}"
        )
    return "\n\n".join(parts)


def stream_answer(query: str, chunks: list[dict]):
    client = get_claude_client()
    system = SYSTEM_PROMPT.format(context=build_context(chunks))
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": query}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# UI — SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<h2 style="color:{_BRAND_NAVY};margin:0 0 0.2rem;">🩺 PharmaForceIQ</h2>'
        f'<p style="color:{_BRAND_TEAL};font-weight:600;margin:0 0 1rem;">'
        f"Retrieval Model Comparison Demo</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Embedding Model**")
    selected_model = st.radio(
        "Choose which model powers retrieval:",
        options=list(MODEL_CHOICES.keys()),
        format_func=lambda k: MODEL_CHOICES[k]["label"],
        key="selected_model",
        label_visibility="collapsed",
    )
    st.caption(f"Collection: `{MODEL_CHOICES[selected_model]['collection']}`")

    st.divider()
    st.markdown("**Pipeline Stages**")
    st.success("✅ Medical Literature (Qdrant) — live")
    st.success("✅ HIPAA check (regex + Haiku, fail-closed) — live")
    st.info("🔜 Claims Data (Databricks) — no access to this data source yet")
    st.info("🔜 FDA Drug Labels (OpenFDA) — no access to this data source yet")
    st.info("🔜 Evidence Docs (UpToDate) — no access to this data source yet")
    st.caption(
        "Architecture mirrors the production 4-source parallel retrieval "
        "pipeline — Claims/FDA/Evidence plug into the same `retrieve()` "
        "pattern once those data sources are connected. HIPAA check now "
        "matches production's two-layer design, but fails CLOSED on a "
        "Haiku error instead of production's fail-open behavior."
    )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.warning(
        "**Demo only.** Do not enter real patient-identifiable information. "
        "This app runs the same two-layer PHI check as production (regex + "
        "Claude Haiku), fixed to fail closed on error — but it's a smaller "
        "regex pattern set than production's, so treat it as a real "
        "safeguard, not a guarantee."
    )

# ---------------------------------------------------------------------------
# UI — MAIN
# ---------------------------------------------------------------------------
st.markdown(
    f'<h1 style="color:{_BRAND_NAVY};margin-bottom:0.3rem;">Clinical AI Assistant</h1>',
    unsafe_allow_html=True,
)
st.caption(
    f"Currently retrieving with **{MODEL_CHOICES[selected_model]['label']}** — "
    f"switch models in the sidebar and re-ask the same question to compare."
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
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta = msg["meta"]
            with st.expander("Details", expanded=False):
                st.markdown(f"**Model:** `{_render_meta['model_repo']}`")
                st.markdown(f"**Collection:** `{_render_meta['collection']}`")
                st.markdown(f"**Response time:** {_render_meta['time']}s")
                st.markdown("**Retrieved passages:**")
                for i, c in enumerate(_render_meta["chunks"], 1):
                    st.markdown(
                        f"`[Source {i}]` **{c['doc_stem']}** — "
                        f"_{c['section_title']}_ — score `{c['score']}`"
                    )

typed = st.chat_input("Ask about clinical guidelines or research findings...")
prompt = typed or st.session_state.pop("pending_prompt", None)

if prompt:
    with st.spinner("Checking for patient-identifying information..."):
        phi_detected = contains_possible_phi(prompt)

    if phi_detected:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "⚠️ This question appears to contain patient-identifying "
                "information (this could be a structured pattern like a "
                "phone number or MRN, or a free-text detail like a patient "
                "name — both are checked). Please rephrase using only "
                "general clinical details, not information about a "
                "specific real patient."
            ),
        })
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner(f"Searching with {MODEL_CHOICES[selected_model]['label']}..."):
            chunks = retrieve(prompt, selected_model)

        if not chunks:
            answer = "No relevant passages were found in the indexed literature for this question."
            st.markdown(answer)
        else:
            answer = st.write_stream(stream_answer(prompt, chunks))

        elapsed = round(time.time() - t0, 1)
        meta = {
            "model_repo": MODEL_CHOICES[selected_model]["repo_id"],
            "collection": MODEL_CHOICES[selected_model]["collection"],
            "time": elapsed,
            "chunks": chunks,
        }
        with st.expander("Details", expanded=False):
            st.markdown(f"**Model:** `{meta['model_repo']}`")
            st.markdown(f"**Collection:** `{meta['collection']}`")
            st.markdown(f"**Response time:** {meta['time']}s")
            st.markdown("**Retrieved passages:**")
            for i, c in enumerate(chunks, 1):
                st.markdown(
                    f"`[Source {i}]` **{c['doc_stem']}** — "
                    f"_{c['section_title']}_ — score `{c['score']}`"
                )

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
