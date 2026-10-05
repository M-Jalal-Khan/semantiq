"""SemantIQ query tools.

Two tools the agent (and you, in the terminal) use:
  - get_schema()       -> LLM-readable description of tables, columns, measures
  - execute_query(sql) -> guarded SQL execution against the DuckDB mirror

Safety is deterministic (no LLM involved):
  - read-only: only SELECT / WITH...SELECT, DDL & DML keywords blocked
  - PII redaction: Surname / CustomerId are stripped from result output
  - row cap: raw row queries auto-limited to 100 rows
"""

from pathlib import Path
import re
import time

import duckdb

DB_PATH = Path(__file__).resolve().parents[2] / "semantiq.duckdb"

BLOCKED_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "copy", "attach", "detach", "install", "load", "pragma", "call",
    "vacuum", "checkpoint",
]
PII_COLUMNS = {"surname", "customerid"}
DEFAULT_ROW_LIMIT = 100

# Business descriptions for every certified measure view.
MEASURE_DESCRIPTIONS = {
    "v_total_customers": "Total customers",
    "v_churned_customers": "Customers who churned (Exited=1)",
    "v_retained_customers": "Customers retained (Exited=0)",
    "v_churn_rate": "Churn rate = churned / total",
    "v_retention_rate": "Retention rate = 1 - churn rate",
    "v_total_balance": "Total account balance (EUR)",
    "v_balance_at_risk": "Balance held by churned customers (EUR)",
    "v_share_balance_at_risk": "Share of total balance held by churned customers",
    "v_avg_balance_per_customer": "Average balance per customer (EUR)",
    "v_avg_credit_score": "Average credit score",
    "v_avg_credit_score_churned": "Average credit score of churned customers",
    "v_avg_credit_score_retained": "Average credit score of retained customers",
    "v_avg_age": "Average age",
    "v_avg_age_churned": "Average age of churned customers",
    "v_avg_tenure_years": "Average tenure in years",
    "v_avg_products": "Average number of products held",
    "v_avg_salary": "Average estimated salary (EUR)",
    "v_active_member_share": "Share of customers who are active members",
}

DIMENSION_VALUES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
    "age_group": ["18-29", "30-39", "40-49", "50-59", "60+"],
    "credit_band": ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"],
    "tenure_group": ["New (0-2 yrs)", "Growing (3-5 yrs)", "Established (6-8 yrs)", "Loyal (9-10 yrs)"],
    "balance_band": ["Zero balance", "Low (< €50k)", "Medium (€50k-100k)", "High (€100k-150k)", "Very high (€150k+)"],
    "churn_label": ["Churned", "Retained"],
    "Exited": ["0 = retained", "1 = churned"],
}


def _connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python -m semantiq.db.build")
    return duckdb.connect(str(DB_PATH), read_only=True)


def validate_sql(sql: str) -> tuple:
    """Returns (ok: bool, message: str). Pure deterministic checks."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "Empty query."
    code = re.sub(r"--.*?$", "", s, flags=re.M).strip().lower()
    if not (code.startswith("select") or code.startswith("with")):
        return False, "Only SELECT queries are allowed (read-only)."
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", code):
            return False, f"Blocked keyword: {kw.upper()} (read-only access)."
    return True, "OK"


def get_schema() -> str:
    """Human/LLM-readable description of the semantic mirror."""
    con = _connect()
    cols = con.execute("DESCRIBE fact_churn").fetchall()
    lines = ["TABLE fact_churn (10,000 rows, one row per customer):"]
    for name, dtype, *_ in cols:
        lines.append(f"  - {name} ({dtype})")
    lines.append("")
    lines.append("CERTIFIED MEASURE VIEWS (use these for KPIs, not raw math):")
    for view, desc in MEASURE_DESCRIPTIONS.items():
        lines.append(f"  - {view}: {desc}")
    lines.append("")
    lines.append("FILTER VALUES:")
    for col, vals in DIMENSION_VALUES.items():
        lines.append(f"  - {col}: {', '.join(vals)}")
    lines.append("")
    lines.append("RULES: aggregate through measure views or GROUP BY dimensions; "
                 "never output Surname/CustomerId (redacted).")
    con.close()
    return "\n".join(lines)


def execute_query(sql: str) -> dict:
    """Validate, run, and redact. Returns a result dict."""
    ok, msg = validate_sql(sql)
    if not ok:
        return {"ok": False, "error": msg}
    sql_limited = sql.strip().rstrip(";")
    if not re.search(r"\blimit\b", sql_limited, re.I):
        sql_limited += f" LIMIT {DEFAULT_ROW_LIMIT}"
    con = _connect()
    try:
        t0 = time.perf_counter()
        rel = con.execute(sql_limited)
        cols = [d[0].lower() for d in rel.description]
        rows = rel.fetchall()
        ms = (time.perf_counter() - t0) * 1000
        # PII redaction: drop PII columns from the output
        keep = [i for i, c in enumerate(cols) if c not in PII_COLUMNS]
        cols = [cols[i] for i in keep]
        rows = [[r[i] for i in keep] for r in rows]
        return {"ok": True, "columns": cols, "rows": rows,
                "row_count": len(rows), "elapsed_ms": round(ms, 1)}
    except Exception as e:  # noqa: BLE001 - surface DB errors to the caller
        return {"ok": False, "error": f"SQL error: {e}"}
    finally:
        con.close()


if __name__ == "__main__":
    # Self-test: prints schema + runs sample queries incl. safety checks.
    print(get_schema())
    print("=" * 60)
    tests = [
        "SELECT * FROM v_churn_rate",
        ("SELECT age_group, COUNT(*) FILTER (WHERE Exited=1) * 1.0 / COUNT(*) AS cr "
         "FROM fact_churn GROUP BY 1 ORDER BY 2 DESC"),
        "SELECT Surname FROM fact_churn LIMIT 5",
        "DROP TABLE fact_churn",
        "SELECT * FROM fact_churn",
    ]
    for q in tests:
        r = execute_query(q)
        print(f"Q: {q[:70]}")
        if r["ok"]:
            print(f"   -> columns={r['columns']} rows={r['row_count']} "
                  f"({r['elapsed_ms']}ms)")
            for row in r["rows"][:6]:
                print(f"      {row}")
        else:
            print(f"   -> BLOCKED/ERROR: {r['error']}")
        print()
