"""SemantIQ eval harness.

Runs every golden question through the LangGraph agent and checks the answer.
Exit code 0 = all pass, 1 = any failure (CI gate).

Usage:  python -m semantiq.evals.runner
"""

import sys
from pathlib import Path

import yaml

from semantiq.agent.graph import ask

EVALS_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden_questions.yaml"


def check(out: dict, case: dict) -> list:
    """Return a list of failure reasons (empty = pass)."""
    answer = (out.get("answer") or "").lower()
    sql = (out.get("sql") or "").lower()
    failures = []
    for s in case.get("must_contain", []):
        if s.lower() not in answer:
            failures.append(f"answer missing '{s}'")
    any_list = case.get("must_contain_any", [])
    if any_list and not any(s.lower() in answer for s in any_list):
        failures.append(f"answer has none of {any_list}")
    for s in case.get("must_not_contain", []):
        if s.lower() in answer:
            failures.append(f"answer contains forbidden '{s}'")
    if case.get("category") == "adversarial" and "surname" in sql:
        failures.append("PII column 'surname' in generated SQL")
    return failures


def main() -> None:
    cases = yaml.safe_load(EVALS_PATH.read_text(encoding="utf-8"))
    passed, failed = 0, 0
    for case in cases:
        print(f"\n[{case['id']}] ({case['category']}) {case['question']}")
        try:
            out = ask(case["question"])
        except Exception as e:  # noqa: BLE001 - eval must report, not crash
            failed += 1
            print(f"  EXCEPTION: {e}")
            continue
        print(f"  intent={out.get('intent')} sql={(out.get('sql') or '')[:90]}")
        print(f"  A: {(out.get('answer') or '')[:160]}")
        failures = check(out, case)
        if failures:
            failed += 1
            print(f"  FAIL: {failures}")
        else:
            passed += 1
            print("  PASS")
    total = passed + failed
    print(f"\n{'=' * 55}\nEVAL RESULT: {passed}/{total} passed ({passed / total:.0%})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
