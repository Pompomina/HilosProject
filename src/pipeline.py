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

