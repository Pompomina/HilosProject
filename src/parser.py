"""
Part 1: Data Parsing & Normalization

Loads hilos_last_library_synthetic.xlsx and produces a ParsedData dataclass
containing a merged DataFrame and grading increments dict.

Design choice: DataFrame output (vs. JSON) because it supports vectorized
filtering, comparison, and groupby operations needed by query_tools.py.
safe_float() is the single source of truth for numeric coercion across the project.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
