
## Why this exists

Business users can't self-serve from BI tools: every ad-hoc question waits on an
analyst, or gets misread off a dashboard. Generic chatbots "answer" from thin air.
SemantIQ only answers from **certified measures** — the governed logic behind the
dashboard — and shows its work: the measure used, the SQL run, the rows and the time.

## The demo model

Bank-customer churn: **10,000 customers**, star schema (1 fact + 7 dimensions),
**18 certified measure views** mirrored from DAX. The same model powers a
[Power BI dashboard](https://muhammadjalalkhan.site) — dashboard and agent answer
from identical logic.

Validated anchor numbers the agent reproduces exactly:

| Segment | Churn rate |
|---|---|
| Overall | 20.37% |
| Age 50–59 | 56.0% |
| Germany | 32.4% |
| Age 50–59 × Germany | 70.0% |

## Architecture

| Layer | Choice | Why |
|---|---|---|
| Semantic mirror | DuckDB | Local-first, zero cost, mirrors the star schema 1:1 |
| Agent | LangGraph | Planner → writer → validator → executor → synthesizer |
| LLM | Gemini (free tier) | Reasoning on a zero budget; Ollama as the fully-local path |
| UI | Streamlit | Streaming chat, live agent trace, eval runner, architecture diagram |
| Evals | 12 golden questions | The quality gate for prompt changes |
| Observability | Per-node trace | Latency + tokens per node, visible live in the UI |

The executor is abstracted: DuckDB today, live Power BI dataset via REST
`executeQueries` when a paid workspace is available — no agent redesign needed.

## Governance (the senior bits)

- **Certified measures only** — the agent aggregates through allowlisted measure
  views, never invents KPI math
- **Read-only** — the validator rejects anything that isn't a SELECT
- **No PII** — `Surname` / `CustomerId` never leave the database
- **Cited answers** — every answer carries a "Show your work" expander: intent,
  SQL, rows returned, query time
- **Eval-gated** — 12 golden questions: **11/12 (92%)**. Every factual, breakdown
  and comparison question passes; all three adversarial attacks (PII extraction,
  instruction override / data deletion, out-of-scope request) are safely refused.
  The one miss was a free-tier API quota flake, manually verified correct.
- **Raw reads capped** at 100 rows

## Quickstart

```bash
pip install -r requirements.txt
python -m semantiq.db.build      # build the DuckDB semantic mirror (20/20 gate)
python -m semantiq.agent.cli     # ask questions in the terminal
python -m streamlit run semantiq/ui/app.py   # chat UI with live trace
python -m semantiq.evals.runner  # 12 golden questions

Copy `.env.example` to `.env` and add your Gemini API key
(free: https://aistudio.google.com/apikey).
`.env` is gitignored — never commit it.

## UI

Three tabs: **Chat** (streaming answers, live per-node trace panel, starter chips,
"Show your work" expanders) · **Eval Harness** (run the 12-question gate with one
click) · **Architecture** (the LangGraph pipeline diagram).

## Roadmap

- [x] Phase 0 — semantic mirror + 20/20 correctness gate (DuckDB == Power BI numbers)
- [x] Phase 1 — query tool + terminal Q&A (read-only enforced, PII redacted)
- [x] Phase 2 — LangGraph agent (5 nodes, 1 retry on SQL error)
- [x] Phase 3 — eval harness (11/12; all adversarial attacks refused)
- [x] Phase 4 — Streamlit chat UI (trace panel, eval runner, architecture diagram)
- [ ] Phase 5 — docs, demo video, polish
- [ ] Live Power BI REST executor (when a paid workspace is available)
