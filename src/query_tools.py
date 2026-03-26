"""
Part 2: Deterministic query functions.

All functions operate on a ParsedData instance (real DataFrame + grading dict).
Every function returns a structured result dict — never a raw None — so the
LLM always has enough context to produce a helpful message.

Result shapes:
  {"status": "ok", "value": <float>, "is_estimated": False}
  {"status": "ok", "value": <float>, "is_estimated": True, "grading_note": "..."}
  {"status": "not_found", "reason": "size_unavailable", "available_sizes": [...]}
  {"status": "not_found", "reason": "last_unknown", "closest_matches": [...]}
  {"status": "not_found", "reason": "no_matches", "message": "..."}
  {"status": "ok", "results": [...]}   ← for list-return tools
"""

from __future__ import annotations

from typing import Any

from src.parser import DIMENSION_FIELDS, ParsedData, safe_float


# Helpers
def _row_to_dict(row) -> dict[str, Any]:
    """Convert a DataFrame row to a plain dict, coercing floats via safe_float."""
    d = row.to_dict()
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = safe_float(v)
    return d


def _available_sizes(data: ParsedData, last_code: str) -> list[float]:
    df = data.lasts_df
    mask = df["last_code"].str.upper() == last_code.upper()
    sizes = sorted(df.loc[mask, "size_us"].dropna().tolist())
    return sizes


def _lookup_row(data: ParsedData, last_code: str, size_us: float):
    """Return the matching DataFrame row or None."""
    df = data.lasts_df
    mask = (df["last_code"].str.upper() == last_code.upper()) & (df["size_us"] == size_us)
    rows = df[mask]
    return rows.iloc[0] if len(rows) > 0 else None


# Tool 1: lookup_dimension
def lookup_dimension(
    data: ParsedData,
    last_code: str,
    size_us: float,
    dimension: str,
) -> dict[str, Any]:
    """Return the exact value for a single dimension of a specific last+size.

    Args:
        data:       ParsedData instance from load_data().
        last_code:  Exact last code string.
        size_us:    US size as a float (e.g. 9.0, 9.5).
        dimension:  Canonical field name (e.g. 'ball_girth').

    Returns:
        {"status": "ok", "last_code": ..., "size_us": ..., "dimension": ...,
         "value": <float | None>, "is_estimated": False}
        or a not_found dict.
    """
    if dimension not in DIMENSION_FIELDS:
        return {
            "status": "error",
            "message": f"Unknown dimension '{dimension}'. Valid fields: {DIMENSION_FIELDS}",
        }

    # Check last exists at all
    all_codes_upper = data.lasts_df["last_code"].str.upper().unique().tolist()
    if last_code.upper() not in all_codes_upper:
        return {
            "status": "not_found",
            "reason": "last_unknown",
            "message": f"No last with code '{last_code}'.",
        }

    row = _lookup_row(data, last_code, safe_float(size_us))
    if row is None:
        return {
            "status": "not_found",
            "reason": "size_unavailable",
            "last_code": last_code,
            "requested_size": size_us,
            "available_sizes": _available_sizes(data, last_code),
        }

    value = safe_float(row[dimension])
    return {
        "status": "ok",
        "last_code": row["last_code"],
        "size_us": row["size_us"],
        "dimension": dimension,
        "value": value,
        "is_estimated": False,
    }


# Tool 2: compare_lasts
def compare_lasts(
    data: ParsedData,
    size_us: float,
    dimension: str,
    find: str = "max",
) -> dict[str, Any]:
    """Return all lasts at a given size ranked by a dimension value.

    Args:
        data:       ParsedData instance.
        size_us:    US size to compare across.
        dimension:  Canonical field name.
        find:       "max" or "min" — which extreme to highlight.

    Returns:
        {"status": "ok", "size_us": ..., "dimension": ..., "find": ...,
         "results": [{"last_code": ..., "value": ..., "gender": ..., "category": ...}, ...],
         "winner": <last_code of max/min>}
    """
    if dimension not in DIMENSION_FIELDS:
        return {"status": "error", "message": f"Unknown dimension '{dimension}'."}

    df = data.lasts_df
    size_val = safe_float(size_us)
    subset = df[df["size_us"] == size_val].copy()

    if subset.empty:
        all_sizes = sorted(df["size_us"].dropna().unique().tolist())
        return {
            "status": "not_found",
            "reason": "size_unavailable",
            "requested_size": size_us,
            "available_sizes": all_sizes,
        }

    subset = subset[["last_code", "gender", "category", dimension]].copy()
    subset[dimension] = subset[dimension].apply(safe_float)
    subset = subset.dropna(subset=[dimension])
    subset = subset.sort_values(dimension, ascending=(find == "min"))

    results = []
    for _, row in subset.iterrows():
        results.append({
            "last_code": row["last_code"],
            "gender": row.get("gender"),
            "category": row.get("category"),
            "value": row[dimension],
        })

    winner = results[0]["last_code"] if results else None
    return {
        "status": "ok",
        "size_us": size_val,
        "dimension": dimension,
        "find": find,
        "results": results,
        "winner": winner,
    }


# Tool 3: filter_lasts
def filter_lasts(
    data: ParsedData,
    gender: str | None = None,
    size_us: float | None = None,
    dimension: str | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict[str, Any]:
    """Return lasts matching optional gender, size, and dimension range filters.

    Args:
        data:       ParsedData instance.
        gender:     "MENS" or "WOMENS" (case-insensitive).
        size_us:    Filter to this exact size.
        dimension:  Field to apply min_val/max_val filter on.
        min_val:    Lower bound (inclusive).
        max_val:    Upper bound (inclusive).

    Returns:
        {"status": "ok", "results": [...], "count": N}
        or {"status": "not_found", "reason": "no_matches", ...}
    """
    if dimension is not None and dimension not in DIMENSION_FIELDS:
        return {"status": "error", "message": f"Unknown dimension '{dimension}'."}

    df = data.lasts_df.copy()

    if gender is not None:
        df = df[df["gender"].str.upper() == gender.upper()]

    if size_us is not None:
        df = df[df["size_us"] == safe_float(size_us)]

    if dimension is not None:
        df[dimension] = df[dimension].apply(safe_float)
        if min_val is not None:
            df = df[df[dimension] >= min_val]
        if max_val is not None:
            df = df[df[dimension] <= max_val]

    if df.empty:
        return {
            "status": "not_found",
            "reason": "no_matches",
            "message": "No lasts matched the given filters.",
            "filters_applied": {
                "gender": gender,
                "size_us": size_us,
                "dimension": dimension,
                "min_val": min_val,
                "max_val": max_val,
            },
        }

    cols = ["last_code", "gender", "category", "size_us"]
    if dimension:
        cols.append(dimension)
    results = []
    for _, row in df[cols].iterrows():
        entry = {c: row[c] for c in cols}
        if dimension:
            entry[dimension] = safe_float(entry[dimension])
        results.append(entry)

    return {"status": "ok", "results": results, "count": len(results)}


# Tool 4: estimate_graded
def estimate_graded(
    data: ParsedData,
    last_code: str,
    known_size: float,
    target_size: float,
    dimension: str,
) -> dict[str, Any]:
    """Estimate a dimension value at target_size using linear grading.

    Formula: estimated = known_value + (target_size - known_size) * grading_rate

    NOTE: Only use this tool when the target size does NOT exist in the dataset.
    If the target size already exists, call lookup_dimension instead.

    Args:
        data:        ParsedData instance.
        last_code:   Exact last code.
        known_size:  The size with a direct measurement.
        target_size: The size to estimate for.
        dimension:   Canonical field name.

    Returns:
        {"status": "ok", "value": <float>, "is_estimated": True, "grading_note": "..."}
        or not_found dict.
    """
    if dimension not in DIMENSION_FIELDS:
        return {"status": "error", "message": f"Unknown dimension '{dimension}'."}

    # Look up the known value
    lookup = lookup_dimension(data, last_code, safe_float(known_size), dimension)
    if lookup["status"] != "ok":
        return lookup

    known_value = lookup["value"]
    if known_value is None:
        return {
            "status": "not_found",
            "reason": "missing_value",
            "message": f"'{dimension}' is None for {last_code} size {known_size}.",
        }

    rate = data.grading.get(dimension)
    if rate is None:
        return {
            "status": "error",
            "message": f"No grading rate available for dimension '{dimension}'.",
        }

    estimated = known_value + (safe_float(target_size) - safe_float(known_size)) * rate
    delta = safe_float(target_size) - safe_float(known_size)
    sign = "+" if delta >= 0 else ""
    note = (
        f"{known_value:.1f} {sign}{delta:+.1f} size × {rate} mm/size = {estimated:.1f} mm"
        f" (graded from size {known_size} to size {target_size})"
    )

    return {
        "status": "ok",
        "last_code": last_code,
        "target_size": safe_float(target_size),
        "known_size": safe_float(known_size),
        "dimension": dimension,
        "value": round(estimated, 2),
        "is_estimated": True,
        "grading_rate": rate,
        "grading_note": note,
    }


# Tool 5: list_lasts
def list_lasts(data: ParsedData) -> dict[str, Any]:
    """Return all unique last codes with their metadata.

    Use this tool only when the last code is genuinely unknown.
    Do NOT call it when the last code is already present in the query.

    Returns:
        {"status": "ok", "lasts": [{"last_code": ..., "gender": ..., "category": ...,
         "vendor": ..., "available_sizes": [...]}, ...], "count": N}
    """
    df = data.lasts_df
    meta_cols = ["last_code", "gender", "category", "vendor"]
    available_meta = [c for c in meta_cols if c in df.columns]

    result_lasts = []
    for code in sorted(df["last_code"].dropna().unique()):
        mask = df["last_code"] == code
        row = df[mask].iloc[0]
        entry = {c: row[c] for c in available_meta}
        entry["available_sizes"] = sorted(
            df.loc[mask, "size_us"].dropna().tolist()
        )
        result_lasts.append(entry)

    return {"status": "ok", "lasts": result_lasts, "count": len(result_lasts)}


# Tool registry for pipeline (tool name → function)
TOOL_FUNCTIONS = {
    "lookup_dimension": lookup_dimension,
    "compare_lasts": compare_lasts,
    "filter_lasts": filter_lasts,
    "estimate_graded": estimate_graded,
    "list_lasts": list_lasts,
}

# Anthropic tool schemas (used in pipeline.py)
TOOL_SCHEMAS = [
    {
        "name": "lookup_dimension",
        "description": (
            "Look up the exact measured value for a specific dimension of a last "
            "at a given size. Use this tool for direct lookup queries. "
            "IMPORTANT: Prefer this over estimate_graded when the requested size exists in the dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "last_code": {"type": "string", "description": "Exact last code, e.g. 'HS010125ML-1'"},
                "size_us": {"type": "number", "description": "US size as a number, e.g. 9 or 9.5"},
                "dimension": {
                    "type": "string",
                    "enum": DIMENSION_FIELDS,
                    "description": "Canonical dimension field name",
                },
            },
            "required": ["last_code", "size_us", "dimension"],
        },
    },
    {
        "name": "compare_lasts",
        "description": (
            "Compare all lasts at a given size by a specific dimension, returning "
            "them ranked from highest to lowest (or lowest to highest). "
            "Use for 'which last has the widest/narrowest...' queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "size_us": {"type": "number", "description": "US size to compare across"},
                "dimension": {"type": "string", "enum": DIMENSION_FIELDS},
                "find": {
                    "type": "string",
                    "enum": ["max", "min"],
                    "description": "'max' to find the largest, 'min' to find the smallest",
                },
            },
            "required": ["size_us", "dimension", "find"],
        },
    },
    {
        "name": "filter_lasts",
        "description": (
            "Filter lasts by gender, size, and/or a dimension value range. "
            "Use for 'list all men's lasts with...' or range filter queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gender": {
                    "type": "string",
                    "enum": ["MENS", "WOMENS"],
                    "description": "Filter by gender",
                },
                "size_us": {"type": "number", "description": "Filter to this exact US size"},
                "dimension": {
                    "type": "string",
                    "enum": DIMENSION_FIELDS,
                    "description": "Dimension to apply range filter on",
                },
                "min_val": {"type": "number", "description": "Minimum value (inclusive)"},
                "max_val": {"type": "number", "description": "Maximum value (inclusive)"},
            },
            "required": [],
        },
    },
    {
        "name": "estimate_graded",
        "description": (
            "Estimate a dimension value at a target size using linear grading rules, "
            "starting from a known measured size. "
            "ONLY use this when the target size does NOT exist in the dataset. "
            "Always prefer lookup_dimension when the size is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "last_code": {"type": "string", "description": "Exact last code"},
                "known_size": {"type": "number", "description": "The size with a direct measurement"},
                "target_size": {"type": "number", "description": "The size to estimate"},
                "dimension": {"type": "string", "enum": DIMENSION_FIELDS},
            },
            "required": ["last_code", "known_size", "target_size", "dimension"],
        },
    },
    {
        "name": "list_lasts",
        "description": (
            "List all available last codes with their metadata and size ranges. "
            "Use ONLY when the user has not specified a last code and you need to "
            "discover what is available. Do NOT call this when a last code is "
            "already present in the query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
