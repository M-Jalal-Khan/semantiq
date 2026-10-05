"""SemantIQ LangGraph agent — with per-node trace (timing + tokens).

Graph:
    planner -> writer -> validator -> executor -> synthesizer
                              ^________| (one retry: error fed back to writer)

Every node appends to state["trace"]: {node, ms, tok_in, tok_out}.
The UI renders this as the live Agent Trace.

Run self-test:  python -m semantiq.agent.graph
"""

import operator
import os
import re
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

from semantiq.db.tools import execute_query, get_schema, validate_sql

MODEL_NAME = "gemini-3.5-flash-lite"

WRITER_INSTRUCTIONS = """You translate natural-language business questions into DuckDB SQL.

You will receive the database schema. Rules:
1. Output ONLY the SQL query. No explanations, no markdown fences, no preamble.
2. Single SELECT statement only (WITH ... SELECT is allowed).
3. For headline KPIs (churn rate, totals, averages), prefer the certified measure views (v_*).
4. For breakdowns, GROUP BY the dimension columns on fact_churn
   (Geography, Gender, age_group, credit_band, tenure_group, balance_band, churn_label).
5. Exited: 1 = churned, 0 = retained.
6. Never select Surname or CustomerId.
7. If the question cannot be answered from this data, output exactly: OUT_OF_SCOPE
8. "Churns most/least/highest/lowest" means churn RATE (a ratio), not absolute
   churned-customer counts. Rank segments by rate.
"""

PLANNER_INSTRUCTIONS = """Classify this business question into exactly one intent.
Reply with ONLY the intent name, nothing else.

Intents:
- metric_lookup: asks for a single headline KPI (churn rate, totals, averages)
- breakdown: splits or ranks a metric across segments (by age, by country, most/least...)
- comparison: compares two groups or values head-to-head
- out_of_scope: cannot be answered from churn data (opinions, predictions, unrelated topics)
"""

SYNTH_INSTRUCTIONS = """You present BI query results as a concise natural-language answer.

You receive: the user's question, the SQL that was run, and the result rows.
Rules:
- Lead with the headline number, formatted readably (percentages, euro, commas).
- 2-3 sentences max. Plain text, no markdown tables.
- Briefly name the measure or grouping used.
- If rows are empty, say the data has no matching records.
- Never invent numbers not present in the results.
"""


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    sql: str
    result: dict
    answer: str
    error: str
    retries: int
    last_error: str
    trace: Annotated[list, operator.add]


def _make_model(system_instruction: str):
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME, system_instruction=system_instruction)


def _generate(model, prompt: str) -> tuple:
    """Returns (text, usage). Handles free-tier quota with wait-and-retry."""
    from google.api_core import exceptions

    def _call():
        resp = model.generate_content(prompt)
        usage = {}
        try:
            um = resp.usage_metadata
            usage = {"in": int(um.prompt_token_count or 0),
                     "out": int(um.candidates_token_count or 0)}
        except Exception:  # noqa: BLE001 - usage is best-effort
            pass
        return resp.text.strip(), usage

    try:
        return _call()
    except exceptions.ResourceExhausted as e:
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
        wait = min(int(m.group(1)) + 10, 90) if m else 65
        print(f"  ...free-tier limit hit, waiting {wait}s then retrying...")
        time.sleep(wait)
        return _call()


def _clean_sql(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    while lines and not re.match(r"(?i)^(select|with)\b", lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _trace(node: str, ms: float, usage: dict = None) -> dict:
    u = usage or {}
    return {"trace": [{"node": node, "ms": round(ms, 1),
                       "tok_in": u.get("in", 0), "tok_out": u.get("out", 0)}]}


# ---------------------------------------------------------------- nodes

def planner_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    model = _make_model(PLANNER_INSTRUCTIONS)
    text, usage = _generate(model, state["question"])
    intent = text.strip().lower()
    if intent not in ("metric_lookup", "breakdown", "comparison", "out_of_scope"):
        intent = "breakdown"
    out = {"intent": intent}
    out.update(_trace("planner", (time.perf_counter() - t0) * 1000, usage))
    return out


def writer_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    schema = get_schema()
    extra = ""
    if state.get("last_error"):
        extra = (f"\n\nYour previous SQL failed with this error:\n{state['last_error']}\n"
                 "Fix it. Return ONLY the corrected SQL.")
    model = _make_model(WRITER_INSTRUCTIONS)
    text, usage = _generate(model, f"{schema}\n\nQuestion: {state['question']}{extra}")
    sql = _clean_sql(text)
    out = {"sql": sql, "error": "out_of_scope" if sql.startswith("OUT_OF_SCOPE") else ""}
    if out["error"]:
        out["sql"] = ""
    out.update(_trace("writer", (time.perf_counter() - t0) * 1000, usage))
    return out


def validator_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    ok, msg = validate_sql(state["sql"])
    out = {"error": ""} if ok else {"error": f"validation: {msg}"}
    out.update(_trace("validator", (time.perf_counter() - t0) * 1000))
    return out


def executor_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    res = execute_query(state["sql"])
    out = {"result": res, "error": ""} if res["ok"] else {"result": {}, "error": res["error"]}
    out.update(_trace("executor", (time.perf_counter() - t0) * 1000))
    return out


def retry_prep_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    out = {"retries": state.get("retries", 0) + 1,
           "last_error": state.get("error"), "error": ""}
    out.update(_trace("retry_prep", (time.perf_counter() - t0) * 1000))
    return out


def synthesizer_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    usage: dict = {}
    if state.get("error") == "out_of_scope":
        answer = ("I can't answer that from the churn data. "
                  "Try customers, churn, balances, or segments.")
    elif state.get("error"):
        answer = f"I couldn't run that query safely: {state['error']}"
    else:
        model = _make_model(SYNTH_INSTRUCTIONS)
        res = state["result"]
        prompt = (f"Question: {state['question']}\nSQL run: {state['sql']}\n"
                  f"Columns: {res['columns']}\nRows: {res['rows'][:20]}")
        answer, usage = _generate(model, prompt)
    out = {"answer": answer}
    out.update(_trace("synthesizer", (time.perf_counter() - t0) * 1000, usage))
    return out


# ------------------------------------------------------------- routing

def _route_after_writer(state: AgentState) -> str:
    return "synthesizer" if state.get("error") else "validator"


def _route_after_validator(state: AgentState) -> str:
    return "synthesizer" if state.get("error") else "executor"


def _route_after_executor(state: AgentState) -> str:
    if not state.get("error"):
        return "synthesizer"
    return "retry_prep" if state.get("retries", 0) < 1 else "synthesizer"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("writer", writer_node)
    g.add_node("validator", validator_node)
    g.add_node("executor", executor_node)
    g.add_node("retry_prep", retry_prep_node)
    g.add_node("synthesizer", synthesizer_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "writer")
    g.add_conditional_edges("writer", _route_after_writer,
                            {"synthesizer": "synthesizer", "validator": "validator"})
    g.add_conditional_edges("validator", _route_after_validator,
                            {"synthesizer": "synthesizer", "executor": "executor"})
    g.add_conditional_edges("executor", _route_after_executor,
                            {"synthesizer": "synthesizer", "retry_prep": "retry_prep"})
    g.add_edge("retry_prep", "writer")
    g.add_edge("synthesizer", END)
    return g.compile()


def ask(question: str) -> dict:
    """Run one question through the full graph. Returns the final state."""
    return build_graph().invoke({"question": question, "retries": 0, "trace": []})


if __name__ == "__main__":
    out = ask("What is the overall churn rate?")
    print(f"intent: {out.get('intent')}")
    print(f"answer: {out.get('answer')}")
    print("trace:")
    for t in out.get("trace", []):
        print(f"  {t['node']:<12} {t['ms']:>8.1f}ms  {t['tok_in'] + t['tok_out']} tokens")
