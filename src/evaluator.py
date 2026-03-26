"""
Part 3: Evaluation harness.

Runs the pipeline against the 12 test cases in eval/test_cases.json and
reports per-type and overall accuracy.

  - Reads answer_value from pipeline output directly (no string parsing).
  - Numeric answers: pass if within ±tolerance_mm of ground truth.
  - List answers: pass if correct set of last codes returned.
  - Tool-selection cases: pass if logged tool_called == expected_tool.
  - no_list_lasts cases: pass if list_lasts NOT in tool_called.

Usage:
    python -m src.evaluator
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TEST_CASES_FILE = Path(__file__).parent.parent / "eval" / "test_cases.json"


# Grading helpers
def _grade_numeric(actual: Any, expected: float, tolerance: float) -> bool:
    """Pass if actual is within ±tolerance of expected."""
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _grade_list(actual: Any, expected: list) -> bool:
    """Pass if actual contains exactly the expected set of last codes (order-insensitive)."""
    if not isinstance(actual, list):
        # Might be a single string
        actual = [actual] if actual else []
    actual_set = {str(v).strip() for v in actual}
    expected_set = {str(v).strip() for v in expected}
    return actual_set == expected_set


def _grade_list_subset(actual: Any, expected: list) -> bool:
    """Pass if all expected codes are present in actual (for 'at least' answers)."""
    if not isinstance(actual, list):
        actual = [actual] if actual else []
    actual_set = {str(v).strip() for v in actual}
    expected_set = {str(v).strip() for v in expected}
    return expected_set.issubset(actual_set)


def _grade_case(case: dict, result: dict) -> tuple[bool, str]:
    """Return (passed, reason_str) for a single test case."""
    t = case["type"]
    tool_called = result.get("tool_called")
    answer_value = result.get("answer_value")
    expected_tool = case.get("expected_tool")
    forbidden_tool = case.get("forbidden_tool")
    tolerance = case.get("tolerance_mm") or 1.0
    expected_value = case.get("expected_answer_value")

    # ---- Tool selection test ----
    if t == "tool_selection":
        if tool_called == expected_tool:
            return True, f"Correct tool '{tool_called}' called"
        return False, f"Expected tool '{expected_tool}', got '{tool_called}'"

    # ---- No-list-lasts test ----
    if t == "no_list_lasts":
        if forbidden_tool and tool_called == forbidden_tool:
            return False, f"Forbidden tool '{forbidden_tool}' was called"
        # Also check answer correctness
        if isinstance(expected_value, (int, float)):
            if _grade_numeric(answer_value, expected_value, tolerance):
                return True, f"Correct answer {answer_value} (tool={tool_called}, not list_lasts)"
            return False, f"Answer {answer_value} not within {tolerance}mm of {expected_value}"
        return True, "list_lasts not called"

    # ---- Numeric answer ----
    if isinstance(expected_value, (int, float)):
        if _grade_numeric(answer_value, expected_value, tolerance):
            reason = f"answer_value={answer_value} within ±{tolerance}mm of {expected_value}"
            return True, reason
        return False, f"answer_value={answer_value!r} not within ±{tolerance}mm of {expected_value}"

    # ---- String / single last code answer ----
    if isinstance(expected_value, str):
        actual_str = str(answer_value).strip() if answer_value is not None else ""
        if actual_str == expected_value.strip():
            return True, f"Correct: '{actual_str}'"
        # Also accept if expected is contained in a list result
        if isinstance(answer_value, list) and expected_value in [str(v) for v in answer_value]:
            return True, f"Expected '{expected_value}' found in result list"
        return False, f"Expected '{expected_value}', got '{actual_str!r}'"

    # ---- List answer ----
    if isinstance(expected_value, list):
        if _grade_list(answer_value, expected_value):
            return True, f"Correct set: {answer_value}"
        if _grade_list_subset(answer_value, expected_value):
            return True, f"All expected codes present (superset returned): {answer_value}"
        return False, f"Expected set {expected_value}, got {answer_value!r}"

    return False, f"Unhandled case type '{t}' or expected_value type"


# Main evaluation runner

def run_evaluation(verbose: bool = True) -> dict[str, Any]:
    """Run all test cases and return a summary dict."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.pipeline import Pipeline

    cases = json.loads(TEST_CASES_FILE.read_text(encoding="utf-8"))
    pipeline = Pipeline()

    results_by_type: dict[str, list] = {}
    all_results = []

    for case in cases:
        cid = case["id"]
        t = case["type"]
        question = case["question"]

        if verbose:
            print(f"\n[{cid}/{len(cases)}] [{t.upper()}] {question}")

        result = pipeline.ask(question)

        passed, reason = _grade_case(case, result)

        entry = {
            "id": cid,
            "type": t,
            "question": question,
            "passed": passed,
            "reason": reason,
            "tool_called": result.get("tool_called"),
            "answer_value": result.get("answer_value"),
            "query_type": result.get("query_type"),
            "expected_answer_value": case.get("expected_answer_value"),
            "expected_tool": case.get("expected_tool"),
        }
        all_results.append(entry)
        results_by_type.setdefault(t, []).append(entry)

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} | tool={result.get('tool_called')} | "
                  f"value={result.get('answer_value')!r} | {reason}")

    # Summary
    total = len(all_results)
    passed_total = sum(1 for r in all_results if r["passed"])
    accuracy = passed_total / total if total > 0 else 0.0

    type_summary = {}
    for t, entries in results_by_type.items():
        n = len(entries)
        p = sum(1 for e in entries if e["passed"])
        type_summary[t] = {"passed": p, "total": n, "accuracy": p / n if n > 0 else 0.0}

    summary = {
        "total": total,
        "passed": passed_total,
        "accuracy": accuracy,
        "by_type": type_summary,
        "results": all_results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"OVERALL ACCURACY: {passed_total}/{total} = {accuracy:.0%}")
        print("\nBy query type:")
        for t, s in type_summary.items():
            print(f"  {t:<18} {s['passed']}/{s['total']}  ({s['accuracy']:.0%})")

        print("\n--- FAILURE MODES OBSERVED ---")
        for r in all_results:
            if not r["passed"]:
                print(f"  [{r['id']}] {r['type']}: {r['reason']}")
                print(f"       Q: {r['question'][:80]}")

    return summary


if __name__ == "__main__":
    summary = run_evaluation(verbose=True)
    # Exit with non-zero code if accuracy < 70%
    if summary["accuracy"] < 0.70:
        sys.exit(1)
