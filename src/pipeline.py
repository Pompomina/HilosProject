"""
Part 2: LLM tool-use pipeline.

Orchestrates the full query flow:
  user_query
    → resolver (ambiguity block / normalize)
    → Claude claude-sonnet-4-6 with tool schemas
    → execute Python tool against real DataFrame
    → Claude synthesizes natural language answer + answer_value
    → format_response (presentation layer)
    → log_trace (JSON Lines to logs/traces.jsonl)
    → structured output dict

Also contains format_response() and log_trace() as single functions
(per plan: refactor to separate modules only after end-to-end verification).

Output shape:
  {
    "answer": <str>,
    "answer_value": <float | list | None>,
    "query_type": "lookup" | "comparison" | "filter" | "grading" | "not_found" | "meta",
    "tool_called": <str | None>,
    "tool_result": <dict | None>,
    "error": <str | None>,
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from src.parser import ParsedData, load_data
from src.query_tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from src.resolver import AmbiguityError, Resolver

# Logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
TRACES_FILE = LOGS_DIR / "traces.jsonl"

# System prompt
SYSTEM_PROMPT = """You are an expert assistant for HILOS Studio's footwear last library.
A "last" is a 3D foot-shaped mold that defines shoe geometry. Each last has precise
dimensional measurements across sizes.

CRITICAL RULES — you must follow these without exception:
1. You have NO knowledge of specific numeric values in this dataset.
2. You MUST call a tool before stating any numerical fact.
3. Do NOT guess, infer, or recall dimension values from memory — always use a tool.
4. If no tool result is available, say "I don't have that data" — never fabricate a number.
5. When the requested size exists in the dataset, ALWAYS call lookup_dimension.
   Only call estimate_graded when the exact size is NOT in the dataset.
6. Do NOT call list_lasts when the last code is already specified in the query.

AFTER receiving tool results, synthesize a clear natural language answer. Then append
a JSON block on a new line in this exact format so the evaluator can parse it:
{"query_type": "<lookup|comparison|filter|grading|not_found|meta>", "answer_value": <number, list of last_codes, or null>}

For grading estimates, state the formula used (e.g. "239.8 + (8−9) × 4 = 235.8 mm")
and add the caveat: "This is an estimate based on linear grading rules, not a direct measurement."

For not_found results, list available alternatives (sizes, last codes).
For estimated values, flag them explicitly.

Dimension reference:
- stick_length: Overall last length (mm)
- lbp_length: Last bottom pattern length (mm)
- ball_girth: Forefoot circumference (mm)
- waist_girth: Waist circumference (mm)
- instep_girth: Instep circumference (mm)
- lbp_ball_width: Bottom pattern ball width (mm)
- lbp_heel_width: Bottom pattern heel width (mm)
- heel_height: Heel height at back (mm)
- toe_spring: Toe spring (mm)
"""


# format_response — presentation layer (single function, per plan)
def format_response(
    answer: str,
    query_type: str,
    tool_result: dict | None,
) -> str:
    """Shape the answer text for display.

    For terminal / API use, this returns a formatted string.
    The Streamlit UI calls this same function and further wraps in st widgets.
    """
    if not answer:
        return "(No response generated.)"

    # Strip the embedded JSON block for clean display (it's kept in the
    # structured output dict for programmatic use)
    lines = answer.strip().split("\n")
    display_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{") and "query_type" in stripped:
            continue
        display_lines.append(line)
    clean = "\n".join(display_lines).strip()

    if query_type == "grading" and tool_result and tool_result.get("is_estimated"):
        note = "\n\n⚠ This is an estimate based on linear grading rules, not a direct measurement."
        if note not in clean:
            clean += note

    return clean


# log_trace — structured logging (single function, per plan)
def log_trace(entry: dict) -> None:
    """Append a JSON Lines entry to logs/traces.jsonl."""
    LOGS_DIR.mkdir(exist_ok=True)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with TRACES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# _extract_structured — parse the JSON block Claude appends
def _extract_structured(text: str) -> tuple[str | None, Any]:
    """Return (query_type, answer_value) from the embedded JSON block, or (None, None)."""
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line.startswith("{") and "query_type" in line:
            try:
                d = json.loads(line)
                return d.get("query_type"), d.get("answer_value")
            except json.JSONDecodeError:
                pass
    return None, None


# Pipeline
class Pipeline:
    """Main query pipeline.

    Usage:
        pipeline = Pipeline()          # loads data from default xlsx path
        result = pipeline.ask("What is the ball girth of HS010125ML-1 in size 9?")
        print(result["answer"])
    """

    def __init__(self, xlsx_path: str | Path | None = None):
        # Load .env, falling back to .env.example if .env doesn't exist
        _project_root = Path(__file__).parent.parent
        _env = _project_root / ".env"
        _env_example = _project_root / ".env.example"
        try:
            from dotenv import load_dotenv
            if _env.exists():
                load_dotenv(_env, override=False)
            elif _env_example.exists():
                load_dotenv(_env_example, override=False)
        except ImportError:
            pass

        self._data: ParsedData = load_data(xlsx_path) if xlsx_path else load_data()
        self._resolver = Resolver(
            self._data.lasts_df["last_code"].dropna().unique().tolist()
        )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def ask(self, query: str) -> dict[str, Any]:
        """Run a natural language query through the full pipeline.

        Returns a structured dict (see module docstring for shape).
        Never raises — errors are captured in the "error" field.
        """
        trace: dict[str, Any] = {
            "query": query,
            "resolved_query": query,
            "tool_called": None,
            "tool_args": None,
            "tool_result": None,
            "final_answer": None,
            "answer_value": None,
            "query_type": None,
            "error": None,
        }

        # Step 1: Resolver — ambiguity blocking
        try:
            resolved_query = self._resolve_query(query)
            trace["resolved_query"] = resolved_query
        except AmbiguityError as e:
            msg = e.clarification_message()
            trace["error"] = str(e)
            trace["final_answer"] = msg
            log_trace(trace)
            return {
                "answer": msg,
                "answer_value": None,
                "query_type": "ambiguous",
                "tool_called": None,
                "tool_result": None,
                "error": str(e),
            }
        except ValueError as e:
            trace["error"] = str(e)
            msg = str(e)
            trace["final_answer"] = msg
            log_trace(trace)
            return {
                "answer": msg,
                "answer_value": None,
                "query_type": "error",
                "tool_called": None,
                "tool_result": None,
                "error": str(e),
            }

        # Step 2: First LLM call — let Claude decide which tool to use
        messages = [{"role": "user", "content": resolved_query}]

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            trace["error"] = f"LLM call failed: {e}"
            log_trace(trace)
            return {
                "answer": f"Error contacting LLM: {e}",
                "answer_value": None,
                "query_type": "error",
                "tool_called": None,
                "tool_result": None,
                "error": str(e),
            }

        # Step 3: Tool execution loop
        tool_result = None
        tool_called = None

        while response.stop_reason == "tool_use":
            # Collect ALL tool_use blocks — Claude may call multiple tools per turn.
            # Every tool_use block must have a matching tool_result in the next message.
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results_content = []

            for tool_use_block in tool_use_blocks:
                tool_called = tool_use_block.name
                tool_args = tool_use_block.input

                trace["tool_called"] = tool_called
                trace["tool_args"] = tool_args

                # Execute the Python tool
                fn = TOOL_FUNCTIONS.get(tool_called)
                if fn is None:
                    tool_result = {"status": "error", "message": f"Unknown tool '{tool_called}'"}
                else:
                    try:
                        if tool_called == "list_lasts":
                            tool_result = fn(self._data)
                        else:
                            tool_result = fn(self._data, **tool_args)
                    except Exception as e:
                        tool_result = {"status": "error", "message": str(e)}

                trace["tool_result"] = tool_result
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": json.dumps(tool_result, default=str),
                })

            # Feed all results back to Claude in one user message
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results_content},
            ]

            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

        # Step 4: Extract final answer
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        query_type, answer_value = _extract_structured(final_text)
        trace["query_type"] = query_type
        trace["answer_value"] = answer_value
        trace["final_answer"] = final_text

        formatted = format_response(final_text, query_type or "", tool_result)
        log_trace(trace)

        return {
            "answer": formatted,
            "answer_value": answer_value,
            "query_type": query_type,
            "tool_called": tool_called,
            "tool_result": tool_result,
            "error": None,
        }

    def _resolve_query(self, query: str) -> str:
        """Apply resolver normalizations to the query text.

        Currently does word-level last-code resolution (uppercase normalization).
        Raises AmbiguityError if a partial code is found with multiple matches.
        The resolver is intentionally conservative — it only acts when it's
        confident about the match.
        """
        # Try to find any token that looks like it could be a last code
        # (contains digits and letters, length > 4)
        import re

        tokens = re.findall(r"[A-Za-z0-9][-A-Za-z0-9]*", query)
        resolved = query
        for token in tokens:
            if len(token) >= 5 and any(c.isdigit() for c in token):
                try:
                    canonical = self._resolver.resolve_last_code(token)
                    if canonical.upper() != token.upper():
                        resolved = resolved.replace(token, canonical)
                except AmbiguityError:
                    raise
                except ValueError:
                    pass  # Not a last code — skip
        return resolved

    @property
    def data(self) -> ParsedData:
        return self._data


# CLI entry point for quick testing
def _cli():
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    pipeline = Pipeline()
    queries = [
        "What is the ball girth of HS010125ML-1 in size 9?",
        "Which last has the widest ball width in size 10?",
        "List all men's lasts with a stick length over 290mm in size 11.",
        "If I need a size 8 but only have size 9 data for HS010125ML-1, what would the estimated ball girth be?",
    ]
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer']}")
        print(f"   [tool={result['tool_called']}, type={result['query_type']}, value={result['answer_value']}]")


if __name__ == "__main__":
    _cli()
