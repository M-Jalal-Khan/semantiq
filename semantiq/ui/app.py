"""SemantIQ senior UI — observable AI workbench.

Tabs: Chat (live agent trace) | Eval Harness | Architecture.

Run:  python -m streamlit run semantiq/ui/app.py   (from the project root)
"""

import time
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

from semantiq.agent.graph import build_graph, ask
from semantiq.evals.runner import check as eval_check

st.set_page_config(page_title="SemantIQ", page_icon="🧠", layout="wide")

EVALS_PATH = Path("evals/golden_questions.yaml")

st.markdown("""
<style>
.main .block-container { max-width: 1200px; padding-top: 1.2rem; }
div.stButton > button {
    border-radius: 999px; border: 1px solid #C7D2FE;
    background: #EEF2FF; color: #4338CA; font-weight: 500; font-size: 13px;
}
div.stButton > button:hover { background: #E0E7FF; border-color: #6366F1; }
[data-testid="stChatMessage"] { border-radius: 12px; }
[data-testid="stExpander"] { border: 1px solid #E5E7EB; border-radius: 8px; margin-top: 8px; }
section[data-testid="stSidebar"] { background: #EEF2FF; }
.trace-card { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

STAGES = {
    "planner": "Planning the question",
    "writer": "Writing SQL",
    "validator": "Running safety checks",
    "executor": "Querying the data",
    "retry_prep": "Fixing the SQL",
    "synthesizer": "Writing the answer",
}
STARTERS = [
    "What is the overall churn rate?",
    "Which age group churns the most?",
    "What is the churn rate in Germany?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []


def _words(text: str):
    for w in text.split(" "):
        yield w + " "
        time.sleep(0.015)


def _trace_html(trace) -> str:
    if not trace:
        return ("<div class='trace-card'><p style='color:#9ca3af;font-size:13px;margin:0;'>"
                "Ask a question to watch the agent think — every node, timed.</p></div>")
    rows = []
    for t in trace:
        toks = t.get("tok_in", 0) + t.get("tok_out", 0)
        rows.append(
            "<div style='display:flex;justify-content:space-between;align-items:center;"
            "padding:7px 0;border-bottom:1px solid #eef0f3;font-size:13px;'>"
            f"<span><span style='color:#059669;font-weight:700;'>✓</span> <b>{t['node']}</b></span>"
            f"<span style='color:#6b7280;'>{t['ms']:.0f} ms · {toks} tok</span></div>")
    tot_ms = sum(t["ms"] for t in trace)
    tot_tok = sum(t.get("tok_in", 0) + t.get("tok_out", 0) for t in trace)
    rows.append(
        "<div style='display:flex;justify-content:space-between;padding:10px 0 2px;font-size:13px;'>"
        "<span><b>Total</b></span>"
        f"<span style='color:#4f46e5;font-weight:700;'>{tot_ms/1000:.1f}s · {tot_tok:,} tokens</span></div>")
    return "<div class='trace-card'>" + "".join(rows) + "</div>"


def _work_expander(meta: dict):
    with st.expander("Show your work"):
        st.caption(f"intent: `{meta['intent']}`")
        st.code(meta["sql"] or "(no SQL — refused safely)", language="sql")
        st.caption(f"{meta['rows']} rows · {meta['ms']} ms")


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧠 SemantIQ")
    st.caption("Governed Analytics Agent for Power BI semantic models")
    st.divider()
    st.markdown("**Demo model**")
    st.caption("Bank customer churn · 10,000 customers · 18 certified measures · 7 dimensions")
    st.markdown("**Quality gate**")
    st.caption("Eval harness: 11/12 (92%) · adversarial attacks safely refused")
    st.markdown("**Guardrails**")
    st.caption("Read-only SQL · PII redacted · certified measures only")
    st.divider()
    st.caption("Local-first demo on DuckDB — zero setup. Production path: swap the "
               "executor to the live Power BI dataset via REST API.")

# -------------------------------------------------------------- header
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;">
  <div style="font-size:30px;">🧠</div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#1E1B4B;">SemantIQ</div>
    <div style="font-size:13px;color:#6B7280;">Ask your churn data in plain English — every answer traced to a certified measure</div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12px;color:#059669;font-weight:600;">
    <span style="width:8px;height:8px;border-radius:50%;background:#10B981;display:inline-block;"></span> live
  </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_evals, tab_arch = st.tabs(["💬 Chat", "🧪 Eval Harness", "🏗️ Architecture"])

# ============================================================ TAB: CHAT
with tab_chat:
    col_chat, col_trace = st.columns([2, 1])

    with col_trace:
        st.markdown("#### 🔍 Agent Trace")
        trace_box = st.empty()
        trace_box.markdown(_trace_html(st.session_state.last_trace), unsafe_allow_html=True)
        st.caption("Per-node latency and token usage, captured live from the LangGraph run.")

    with col_chat:
        clicked = None
        if not st.session_state.messages:
            st.caption("Try one:")
            cols = st.columns(3)
            for i, (col, q) in enumerate(zip(cols, STARTERS)):
                if col.button(q, use_container_width=True, key=f"starter_chip_{i}"):
                    clicked = q

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and m.get("meta"):
                    _work_expander(m["meta"])

        question = st.chat_input("Ask about churn, customers, balances, segments...") or clicked
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            graph = build_graph()
            final: dict = {}
            trace_box.markdown(_trace_html([]), unsafe_allow_html=True)
            try:
                with st.status("Thinking…", expanded=False) as status:
                    for chunk in graph.stream({"question": question, "retries": 0, "trace": []}):
                        node = next(iter(chunk))
                        final.update(chunk[node])
                        status.update(label=f"{STAGES.get(node, node)}…")
                        trace_box.markdown(_trace_html(final.get("trace", [])),
                                           unsafe_allow_html=True)
                    status.update(label="Done", state="complete", expanded=False)
            except Exception:  # noqa: BLE001 - friendly error, never a traceback
                with st.chat_message("assistant"):
                    st.error("Something went wrong running the agent. Please try again in a moment.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Something went wrong running the agent. Please try again in a moment.",
                    "meta": None})
                st.stop()

            st.session_state.last_trace = final.get("trace", [])
            answer = final.get("answer") or "I couldn't produce an answer."
            res = final.get("result") or {}
            meta = {"intent": final.get("intent", "-"),
                    "sql": final.get("sql", ""),
                    "rows": res.get("row_count", "-"),
                    "ms": res.get("elapsed_ms", "-")}
            with st.chat_message("assistant"):
                full = st.write_stream(_words(answer))
                _work_expander(meta)
            st.session_state.messages.append({"role": "assistant", "content": full, "meta": meta})

# ============================================================ TAB: EVALS
with tab_evals:
    st.markdown("### 🧪 Eval Harness")
    st.caption("12 golden questions · runs the live agent · the quality gate for prompt changes")
    cases = yaml.safe_load(EVALS_PATH.read_text(encoding="utf-8"))

    st.dataframe(
        [{"id": c["id"], "category": c["category"], "question": c["question"]} for c in cases],
        use_container_width=True, hide_index=True)

    if st.button("▶ Run full eval", type="primary", key="run_eval_btn"):
        results, status_box = [], st.empty()
        prog = st.progress(0)
        for i, c in enumerate(cases):
            status_box.caption(f"Running {c['id']} — {c['question'][:60]}…")
            try:
                out = ask(c["question"])
                fails = eval_check(out, c)
                results.append({"id": c["id"], "category": c["category"],
                                "status": "✅ PASS" if not fails else "❌ FAIL",
                                "answer": (out.get("answer") or "")[:120],
                                "detail": "; ".join(fails)})
            except Exception as e:  # noqa: BLE001 - eval must report, not crash
                results.append({"id": c["id"], "category": c["category"],
                                "status": "💥 ERROR", "answer": "", "detail": str(e)[:120]})
            prog.progress((i + 1) / len(cases))
        st.session_state["eval_results"] = results
        status_box.empty()
        prog.empty()

    if st.session_state.get("eval_results"):
        r = st.session_state["eval_results"]
        passed = sum(1 for x in r if "PASS" in x["status"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed", f"{passed}/{len(r)}")
        c2.metric("Score", f"{passed/len(r):.0%}")
        c3.metric("Failed", f"{len(r)-passed}")
        st.dataframe(r, use_container_width=True, hide_index=True)

# ===================================================== TAB: ARCHITECTURE
with tab_arch:
    st.markdown("### 🏗️ Architecture")
    st.caption("How the agent thinks — every question flows through this pipeline")
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#EEF2FF",
                  color="#6366F1", fontname="Helvetica"];
            edge [fontname="Helvetica", fontsize=10];
            planner -> writer -> validator -> executor -> synthesizer;
            executor -> writer [label="retry ×1", style=dashed, color="#EF4444",
                                fontcolor="#EF4444"];
        }
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Nodes**")
        st.markdown("- **planner** — classifies intent: metric, breakdown, comparison, out-of-scope\n"
                    "- **writer** — generates DuckDB SQL from the schema + certified measures\n"
                    "- **validator** — deterministic safety gate: SELECT-only, no PII, no DROP\n"
                    "- **executor** — runs the query read-only against the semantic mirror\n"
                    "- **synthesizer** — turns rows into a 2–3 sentence business answer")
        st.markdown("**Guardrails**")
        st.markdown("- SQL restricted to `SELECT` / `WITH` — writes are impossible\n"
                    "- `Surname` / `CustomerId` never leave the database\n"
                    "- Raw reads capped at 100 rows\n"
                    "- Certified measure views preferred over raw SQL")
    with c2:
        st.markdown("**Stack**")
        st.markdown("- 🐍 Python · 🦆 DuckDB (semantic mirror)\n"
                    "- 🕸️ LangGraph (orchestration) · 🧠 Gemini (reasoning)\n"
                    "- 🎈 Streamlit (this UI) · 🧪 YAML eval harness")
        st.markdown("**Production path**")
        st.markdown("Today the executor queries the local DuckDB mirror — zero setup, "
                    "anyone can clone and run it. When a paid Power BI workspace is "
                    "available, one function swaps to the REST `executeQueries` endpoint "
                    "and the agent queries the live dataset. Nothing else changes.")
