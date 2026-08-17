# PharmaForceIQ Clinical AI Assistant — Retrieval Model Comparison Demo

A standalone Streamlit app for comparing `BAAI/bge-small-en-v1.5` (current
production model) against `abhinand/MedEmbed-small-v0.1` (medical-tuned)
live, in conversation, side by side — deployable for free on Streamlit
Community Cloud.

**What's actually live:** Medical literature retrieval via Qdrant + Claude
Sonnet synthesis, with a toggle to switch embedding models mid-conversation,
plus a full two-layer HIPAA check (regex + Claude Haiku).

**What's intentionally stubbed (visible in the sidebar, not hidden):**
Claims data (Databricks), FDA labels (OpenFDA), and Evidence docs — these
need data-source access this project doesn't have yet, not a code gap.

---

## v2 Update — Now Using Production's REAL Prompt and REAL Retrieval Logic

Verified line-by-line against the actual production repo (`prompts.py`,
`engine.py`, `formatter.py`, `retriever.py`) before building this:

| | v1 (previous) | v2 (this version) |
|---|---|---|
| System prompt | Simplified, 5 rules | **`prompts.py` copied verbatim from production, zero modifications, zero external dependencies** |
| Retrieval diversity | None — could return 5 chunks of 1 document | **`MAX_CHUNKS_PER_STEM=2` + 4x over-fetch, exact match to `retriever.py`** |
| Context format | Custom | **Matches `format_paper_context()`'s guideline branch exactly** (see note below on why this branch) |
| Synthesis call | `max_tokens=1024`, default temperature | **`max_tokens=2000`, `temperature=0.1` — exact match to `stream_synthesize_answer()`** |
| Provider role/type | Not present | **Real sidebar selectors wired to the real `PROVIDER_ROLE_FRAMING`/`PROVIDER_TYPE_FRAMING` dicts — verified these actually change the resulting prompt** |

### Why the guideline citation branch, not the papers/DOI branch
Production's `format_paper_context()` has two citation styles: a rich
author/year/DOI badge style for `db == "papers"`, and a simpler
topic/section style for guidelines. Our chunk payload (from
`chunk_documents.py`/`embed_and_index.py`) never captured bibliographic
metadata (authors, year, journal, DOI) — only `doc_stem`, `section_title`,
`text`. Using the guideline-style citation is the *correct* match for the
data we actually have, not an approximation — using the papers-style
branch would mean fabricating fake author names, which was never on the
table.

### What This Directly Fixes
Rudra's exact complaint — the app returning 5 chunks of one document
("shallow pointers") instead of synthesizing across sources — is fixed at
its root: the diversity-dedup algorithm guarantees at least 3 distinct
documents contribute to any 5-chunk answer (verified with a test
reproducing the original failure mode: 15 candidates from one document,
3 from another, 2 from a third — output correctly draws from all 3, not
just the dominant one).

---

## v3 Update — Real Citation & Numeric Verification (`citation_verify.py`, `numeric_verify.py`)

Live testing after v2 surfaced a real problem: with the real production
prompt in place, Claude Sonnet fabricated a citation — `[Lin et al., 2024]`
in one response, `[Lin et al., 2019]` in another, for the same underlying
source, neither of which exists anywhere in our retrieved data (our chunk
payload has no author metadata at all). This is exactly the failure mode
production's post-hoc verification layer exists to catch — and v2 didn't
have that layer wired in.

**v3 copies `citation_verify.py` and `numeric_verify.py` from the repo
root verbatim** (both have zero external dependencies — confirmed before
copying) and runs them on every answer after streaming completes:

- **Citation check:** since our data has no author/year metadata,
  `papers_cited=[]` is passed — meaning `citation_verify.py` correctly
  flags **every** `[Author et al]`-style citation the model writes, since
  none can ever be verified against data we don't have. This is the
  correct behavior given our corpus, not a workaround.
- **Numeric check:** run against the actual retrieved chunk text, which we
  DO have — so this performs real, meaningful verification, catching any
  percentage, ratio, or lab value the model states that doesn't actually
  appear in what was retrieved.

**Verified this actually catches the real fabricated citation found in
testing** — reproduced the exact `[Lin et al., 2024]` scenario and
confirmed it gets flagged with the same `⚠️ source not retrieved` badge
production would show, not silently passed through.

Both checks' pass/fail counts are now shown in the "Details" panel on
every answer.

---

## v3.2 Update — Fixing the Root Cause, Not Just Flagging It

Live testing found something more precise than "a fabricated citation":
the DOI Claude generated (`10.1002/ehf2.12501`) was **verified as real** —
it resolves to the actual retrieved paper (*"Malnutrition in Acute Heart
Failure with Preserved Ejection Fraction..."*, ESC Heart Failure, 2019).
But the **author name attached to it was wrong** — Claude wrote "Lin et
al.", while the real first author is "Chien" (confirmed by fetching the
actual publisher page). "Lin" is a real co-author, buried mid-list — not
the conventional lead-author citation. One of the two test runs also got
the year wrong (2024 instead of the real 2019).

This is exactly why the check existed and confirms `citation_verify.py`
was working correctly (not over-flagging) — but v3's response to a
detected problem (append a warning badge, leave the wrong name visible)
wasn't strong enough. Two fixes, both in **our own code**, not modifications
to the copied production files:

**1. Prevention — `build_paper_context()` now prepends an explicit,
data-specific instruction:** since our corpus genuinely has zero
author/year/journal metadata (unlike production's), Claude is told
directly that any author name it produces for these sources would come
from its own memory, not from what was retrieved, and to cite by
`[Source N]` only. This doesn't touch `prompts.py`'s existing (more
general) rule against inventing authors — it adds a second, sharper
instruction appropriate to what our specific data does and doesn't
contain.

**2. Correction — new `correct_unverified_citations()` function,** built
on `citation_verify.verify_citations()`'s real, unmodified detection
logic (same span/surname matching as production), but instead of only
adding a badge *next to* the wrong name, it **replaces the fabricated
attribution entirely** with an honest, visible marker:
`[citation removed — unverified]`. A badge next to "Lin et al., 2024"
still puts a wrong name in front of a clinician; removing it doesn't.

**Verified against the exact real failure case** before shipping:
confirmed the wrong author name no longer appears anywhere in the output
after correction, confirmed the "nothing to correct" branch still doesn't
crash, and confirmed the prevention notice is actually present in the
context sent to Claude.

---


## 1. Deploy to Streamlit Community Cloud (free)

### Step 1 — Push this folder to GitHub
```bash
git init
git add .
git commit -m "Initial commit: retrieval model comparison demo"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
`.gitignore` already excludes `.streamlit/secrets.toml` — your real secrets
will never be committed, even by accident.

### Step 2 — Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.
2. Click **"New app"**.
3. Select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **"Advanced settings"** and paste your secrets (see Step 3 below)
   into the **Secrets** box — do this *before* clicking Deploy, or the app
   will show a clear "Missing secret" error on first load (it fails
   gracefully, not with a crash).
6. Click **Deploy**.

First load takes ~30-60 seconds while both embedding models download and
get cached. After that, switching the toggle is fast — models stay warm
for the life of the running app instance (`st.cache_resource`).

### Step 3 — Secrets (paste into Streamlit Cloud's Secrets box, not into any file in this repo)
```toml
QDRANT_URL = "https://<your-cluster-id>.<region>.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```
See `.streamlit/secrets.toml.example` in this repo for the exact format
(that file has placeholder values only — safe to commit, unlike the real
`secrets.toml`).

---

## 2. Local Development

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with real values
streamlit run app.py
```

---

## 3. What Was Verified Before Delivery

| Check | Result |
|---|---|
| Syntax compiles cleanly | ✅ Both `app.py` and the copied `prompts.py` |
| `prompts.py` has zero external dependencies | ✅ Verified — safe to copy as a standalone file |
| Retrieval diversity-dedup logic | ✅ Tested against the exact original failure mode (15 candidates from one document, 3 from another, 2 from a third) — confirmed output correctly draws from 3 distinct documents, not 1 |
| `build_system_prompt()` imports and runs correctly | ✅ Confirmed R1-R24 rules, role framing, and answer-structure template are all present in the generated prompt |
| Provider role/type selectors actually change the prompt | ✅ Confirmed Cardiologist vs. Oncologist framing and Physician vs. APP framing produce genuinely different prompt text |
| Claims/Evidence instructions correctly excluded | ✅ Confirmed `include_claims_instructions=False` means claims-parsing instructions do NOT appear in the prompt, since that source isn't connected |
| Context formatter output format | ✅ Matches production's `=== CLINICAL KNOWLEDGE EVIDENCE ===` header format exactly, tested with realistic multi-document data |
| `citation_verify.py` / `numeric_verify.py` have zero external dependencies | ✅ Verified before copying — pure stdlib (`re`, `unicodedata`) |
| Citation verification catches the real fabricated citation found in testing | ✅ Reproduced the exact `[Lin et al., 2024]` scenario, confirmed it gets flagged with the correct warning badge |
| Numeric verification catches fabricated statistics | ✅ Tested with a synthetic answer containing 3 fabricated numbers not in the source corpus — all 3 correctly flagged |
| Regex PHI detection logic | ✅ Tested against 6 cases (3 should-flag, 3 should-not) — all passed, including that ordinary clinical questions with numbers (ages, dosages) are correctly NOT flagged |
| Two-layer PHI check (regex + Haiku) | ✅ Tested with mocked Haiku responses across 4 cases: regex catches structured PHI without even calling Haiku, Haiku catches a free-text patient name regex misses, a clean question passes through, and — the critical one — a simulated Haiku API failure fails CLOSED (blocks) rather than production's fail-open behavior |
| Server starts cleanly | ✅ Ran the actual app, confirmed clean startup with the new `prompts.py` import, no errors |
| Missing-secrets handling | ✅ Confirmed it fails gracefully with a clear on-screen message (`st.error` + `st.stop()`), not a raw crash |

**Not yet tested (needs your real credentials, not available in this build
environment):** an actual end-to-end query against live Qdrant + Claude, and
a real (non-mocked) Haiku PHI check call. Both are correct by direct code
inspection and isolated logic testing, but a live run is the real
confirmation. Try it right after deploying and let me know what comes back.

---

## 4. Architecture Notes — Why It's Built This Way

- **Standalone, not importing the production `app/` package.** The
  production codebase's Claims/Databricks/HIPAA modules have dependencies
  (Databricks credentials, a larger HIPAA rule set, etc.) that would
  complicate a free-tier public deploy for no benefit right now, since
  none of that data is connected yet. This app mirrors the production
  *pattern* (sidebar, Details expander, References, sample questions,
  toggle-based model selection) without inheriting dependencies it can't
  use yet.
- **`st.cache_resource` for both embedding models, not just the selected
  one** — switching the toggle doesn't reload from scratch; both models
  can be warm simultaneously if both have been used at least once in the
  running session.
- **Two-layer HIPAA check (regex + Claude Haiku), fail-closed.** Regex
  runs first and skips the Haiku call entirely if it already finds a
  match (no wasted API call). If regex passes clean, Haiku does a
  semantic check for free-text PHI (e.g. a bare patient name) regex can't
  catch. Unlike production's `app/hipaa.py`, this fails CLOSED on any
  Haiku API error — the original architecture review flagged production's
  fail-open behavior as its #1 priority finding, so this demo fixes it
  rather than reproducing it.
- **Simplified system prompt** — production uses an 11-role, 931-line
  prompt with R1-R24 claims-data-interpretation rules. Since claims data
  isn't connected in this demo, those rules don't apply yet; the prompt
  here is scoped to what's actually running (grounded literature
  retrieval + citation).

## 5. Next Steps to Extend This

Claims data and FDA labels each map to a real, add-later integration
point once those data sources are accessible — add a `retrieve_claims()`
or `retrieve_fda()` function following the same shape as `retrieve()`,
and add its results to `build_context()`. Evidence docs would follow the
same pattern against a separate Qdrant collection.
