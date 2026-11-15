# Plan: generation, evaluation, guardrails, API, UI and Azure deployment

Retrieval is finished: parent-child chunking, Hybrid search (dense + BM25), then bge reranking. The README has the
evaluation behind each choice. This plan covers everything after retrieval.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM (answers) | Azure OpenAI deployment `gpt-5.4-mini` (model version 2026-03-17) | `gpt-4o-mini` is deprecated and can't be deployed; supports structured JSON output |
| LLM (judge) | The same `gpt-5.4-mini` | One deployment to manage. It may rate its own answers too kindly, which is checked against hand labels |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1,536 dimensions), **in the same resource**, using the same endpoint and key | Already deployed; managed, so no model in the container |
| API style | Azure's **v1 API** (`<endpoint>/openai/v1/`, no api-version), using the `openai` SDK's `OpenAI` client | The only form `gpt-5.4-mini` answered on; versioned calls returned 404 |
| Search | Azure AI Search, **Free tier** | Built-in hybrid (BM25 + vector, fused with RRF); Basic tier (~$75/month) would use up the credit |
| What gets indexed | **The 4,723 parents at 1,536 dimensions** (~30 MB), index `kjv-parents` | Fits the free tier's storage; all 14,618 children at full size (~90 MB) would not |
| Tracing and evaluation | Langfuse Cloud, EU region, Hobby plan | Nothing to host |
| Semantic ranker | Assumed **not available** on the free tier unless a test shows it is | Rerank with local bge instead |
| API | FastAPI | |
| Interface | Streamlit, built to production standard | |
| Hosting | Docker → Azure Container Registry → Azure Container Apps, with keys in Key Vault | Scales to zero; has a monthly free allowance |
| Budget | Azure free trial, about $90, **expires 6 December 2026** | Deployment must work before then; see the timeline |
| Human labels | The user hand-labels about 20 answers | To calibrate the judge |

Changing the deployment is one line in `.env`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, so another model can be compared later.

### What gpt-5.4-mini accepts (tested on the deployment, 15 Nov 2026)

| Parameter | Result |
|---|---|
| `max_tokens` | Rejected: use `max_completion_tokens` |
| `reasoning_effort` | `none` (default, no reasoning tokens), `minimal`, `low`, `medium`, `high`, `xhigh` |
| `temperature`, `top_p` | Allowed only while `reasoning_effort` is `none`; with reasoning on, only `temperature=1` |
| JSON-schema structured output, system message, `seed` | Work |

`app/config.py` has `chat_options()`, which builds these settings and refuses the combinations the model rejects.

Starting settings:
- **Answers:** `reasoning_effort="none"` with `temperature=0`, so the same question gets a repeatable answer.
- **Judge:** `reasoning_effort="low"`, which does a little reasoning before each label. Temperature must be left at 1 when
  reasoning is on.

Phase 4 measures whether more reasoning effort improves faithfulness or judge agreement enough to justify the extra
time and cost.

## Request flow

```
Streamlit UI ──HTTPS──► FastAPI /ask
                          │
  ① Input guardrails ─────┤ blocked → polite refusal
  ② Retrieval: embed question → AI Search hybrid → rerank → top 5 passages
  ③ Retrieval guardrails ─┤ relevance too low → abstain
  ④ Generation: JSON answer, every sentence cites a passage and quotes it
  ⑤ Output guardrails ────┤ quote not in passage / not grounded → retry once, then abstain
                          ▼
              answer + cited verses → Streamlit (thumbs up/down → Langfuse)

Around everything: operational guardrails (rate limit, timeouts, cost cap) and Langfuse tracing.
```

## Phases

### 0. Setup and secrets
- `.env` holds the keys (git-ignored); `.env.example` lists the names (committed).
- `app/config.py` reads the settings in one place and names any missing setting in its error. Any error text from a
  service has secret values masked before it is shown.
- `python -m app.main check` makes one small call to each service and prints only OK/FAIL, deployment names and dimensions.
- Create the free AI Search service and a Langfuse Cloud project (EU).

### 1. Azure retrieval, measured again
- `app/embeddings.py`: embed the parents with `text-embedding-3-small`, cached in `data/embeddings/`.
- `app/azure_search.py`: create the index (parent text, book, chapter, verse range, heading, 1,536-dim vector), upload,
  and search. Results come back in the same shape as the local retrievers.
- Re-run the 54-question evaluation (P@5, R@5, MRR, nDCG@5), compared with today's best, local Hybrid + bge (MRR 0.945):
  - AI Search vector only
  - AI Search hybrid
  - AI Search hybrid + local bge
  - AI Search hybrid + semantic ranker, only if a test call shows the free tier allows it (assumed not)
- The scores decide whether bge stays. Dropping it would make the image smaller and save about 3 s per question.
- New README and notebook section: "Local vs Azure retrieval".

### 2. Generation (`app/generation.py`)
- The v1 `OpenAI` client from `app/config.py`, with settings from `chat_options()` and structured output:
  `{"answer_parts": [{"text", "citations": [{"passage_id", "quote"}]}], "abstained"}`.
- Code checks that every quote appears word for word in the passage it cites.
- **Injection isolation:** instructions go only in the system message; the passages and the question go in marked blocks,
  and the system message says their contents are never instructions.
- **Abstention in two layers:** a relevance threshold before the LLM, and an `abstained` flag from the LLM.
- New CLI command: `python -m app.main ask "question"`. Tests use a fake client.

### 3. Langfuse tracing (`app/tracing.py`)
- `langfuse.openai.OpenAI`, a drop-in replacement for the v1 client, records every LLM call (tokens, cost); `@observe` records the other steps.
- One trace per question. The test sets are Langfuse datasets, so runs can be compared.

### 4. Generation evaluation (`app/gen_evaluation.py`, `app/judge.py`)

| Metric | Method |
|---|---|
| Answer relevancy | Judge, 1–5 |
| Faithfulness (groundedness) | Claims labelled Supported / Partially supported / Not supported / Contradicted; score = supported ÷ all; any contradicted claim fails the answer |
| Context | P@5, R@5, nDCG@5 + judge score for how relevant the passages are |
| Correctness | Judge compares the answer with the reviewed reference answer, 1–5 |
| Citations | Code: the quote exists and the passage was retrieved. Judge: the quote supports its sentence |

- Reference answers: drafted by Claude, reviewed by the user.
- About 15 unanswerable questions; measure how often it abstains on them, and how often it wrongly abstains on answerable ones.
- Judge calibration: compare the judge with the user's ~20 hand labels.

### 5. Guardrails (`app/guardrails/`)

| Layer | Deterministic | Model-based |
|---|---|---|
| Input | Length limit, injection regex, blocklist | Azure AI Content Safety Prompt Shields, or a `gpt-5.4-mini` classifier |
| Retrieval | Score threshold; results only from the KJV index | LLM checks the passages fit the question |
| Output | Quote verification; cited verses were retrieved; citations required | Fast groundedness check; Content Safety |
| Operational | Rate limit, timeouts, token and cost cap, retries, `/health` | Cost and latency anomaly flags from Langfuse |
| Human oversight | Flagged and abstained answers go to the Langfuse review queue | Thumbs up/down; borderline judge scores go to the queue |

- Score deterministic only, model-based only, and defense in depth (rules first, then the model check) on a ~60-input
  test set: bad inputs caught, good inputs wrongly blocked, time added, cost added. Pick the winner and write up the reasons.

### 6. FastAPI (`app/api.py`)
- Endpoints: `POST /ask`, `POST /search`, `POST /feedback`, `GET /health`.
- Pydantic models, clients loaded once at startup, guardrail middleware.
- Tests use `TestClient` and fake clients.

### 7. Streamlit (`ui/streamlit_app.py`)
- Calls only the API.
- Chat layout with answers that appear as they're generated, numbered citation markers, and expandable verse cards with the quoted words highlighted.
- Distinct messages for abstentions and blocked input; friendly errors with a retry button; `st.feedback` linked to the trace.
- Sidebar: example questions and a "show passages" toggle.
- Custom theme; works on phones. Tests use `AppTest`.

### 8. Docker
- `api` and `ui` images, plus `docker-compose.yml` for running locally. No keys in the images.

### 9. Azure deployment
- ACR Basic (~$5/month), two Container Apps (free monthly allowance), Key Vault with managed identity, AI Search Free,
  Content Safety Free.
- Optional: a GitHub Actions workflow that builds and deploys on every push to `main`.

Every phase ends with tests passing, README and notebook sections, and a commit.

## Timeline (credit expires 6 December 2026)

| Dates | Phases |
|---|---|
| 14–17 Nov | 0, 1 |
| 18–22 Nov | 2, 3, 4 (reference answers reviewed and hand labels done in this window) |
| 23–26 Nov | 5, 6 |
| 27–30 Nov | 7, 8 |
| 1–4 Dec | 9: live on Azure, with a few days to spare |

After the credit ends, Azure asks you to upgrade to pay-as-you-go. The free AI Search tier and the Container Apps
allowance still apply after the upgrade, but ACR and Azure OpenAI usage start costing money.
