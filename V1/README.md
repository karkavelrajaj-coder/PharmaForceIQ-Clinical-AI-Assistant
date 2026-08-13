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
| Syntax compiles cleanly | ✅ |
| Regex PHI detection logic | ✅ Tested against 6 cases (3 should-flag, 3 should-not) — all passed, including that ordinary clinical questions with numbers (ages, dosages) are correctly NOT flagged |
| Two-layer PHI check (regex + Haiku) | ✅ Tested with mocked Haiku responses across 4 cases: regex catches structured PHI without even calling Haiku, Haiku catches a free-text patient name regex misses, a clean question passes through, and — the critical one — a simulated Haiku API failure fails CLOSED (blocks) rather than production's fail-open behavior |
| Citation/context-building logic | ✅ Tested with realistic fake chunk data, output format confirmed correct |
| Server starts cleanly | ✅ Ran the actual app, confirmed healthy via Streamlit's `/_stcore/health` endpoint |
| Missing-secrets handling | ✅ Confirmed it fails gracefully with a clear on-screen message (`st.error` + `st.stop()`), not a raw crash — checked server logs for unhandled exceptions, found none |

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
