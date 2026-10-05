"""SemantIQ terminal chat — ask in English, get governed answers.

Usage:
    python -m semantiq.agent.cli

In-chat commands:
    /sql <query>   run raw SQL directly (bypasses the LLM)
    /schema        print the schema the agent sees
    exit | quit    leave
"""

import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from semantiq.db.tools import execute_query, get_schema

# Flash-Lite: 500 free requests/day (vs ~20/day for full Flash models)
MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTIONS = """You translate natural-language business questions into DuckDB SQL.

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


def _get_model():
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Check your .env file.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTIONS)


def _clean_sql(text: str) -> str:
    """Strip markdown fences / preamble, return raw SQL."""
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    while lines and not re.match(r"(?i)^(select|with)\b", lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def nl_to_sql(model, question: str, schema: str) -> str:
    import time
    from google.api_core import exceptions
    prompt = f"{schema}\n\nQuestion: {question}"
    try:
        return _clean_sql(model.generate_content(prompt).text)
    except exceptions.ResourceExhausted as e:
        # parse the retry_delay block specifically; cap the wait at 90s
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
        wait = min(int(m.group(1)) + 10, 90) if m else 65
        print(f"  ...free-tier limit hit, waiting {wait}s then retrying...")
        time.sleep(wait)
        return _clean_sql(model.generate_content(prompt).text)


def _print_result(res: dict) -> None:
    if not res["ok"]:
        print(f"  !! {res['error']}")
        return
    cols, rows = res["columns"], res["rows"]
    if not cols:
        print("  (no columns returned — PII redacted)")
        return
    widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
              for i, c in enumerate(cols)]
    print("  " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows[:20]:
        print("  " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    if len(rows) > 20:
        print(f"  ... ({len(rows)} rows total)")
    print(f"  [{res['row_count']} rows, {res['elapsed_ms']}ms]")


def main() -> None:
    print("SemantIQ terminal chat — ask about churn. Type 'exit' to quit.\n")
    schema = get_schema()
    model = _get_model()
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("bye.")
            break
        if q.lower() == "/schema":
            print(schema)
            continue
        if q.lower().startswith("/sql "):
            sql = q[5:].strip()
            print(f"  sql> {sql}")
        else:
            sql = nl_to_sql(model, q, schema)
            if sql.startswith("OUT_OF_SCOPE"):
                print("  I can't answer that from the churn data. "
                      "Try customers, churn, balances, or segments.")
                continue
            print(f"  sql> {sql}")
        res = execute_query(sql)
        if not res["ok"] and not q.lower().startswith("/sql "):
            # one automatic retry: feed the DB error back to the model
            fix = model.generate_content(
                f"{schema}\n\nThis SQL failed with:\n{res['error']}\n\nSQL:\n{sql}\n\n"
                "Return ONLY the corrected SQL.").text
            sql = _clean_sql(fix)
            print(f"  retry sql> {sql}")
            res = execute_query(sql)
        _print_result(res)
        print()


if __name__ == "__main__":
    main()
