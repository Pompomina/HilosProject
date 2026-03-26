"""
Part 1: Data Parsing & Normalization

Loads hilos_last_library_synthetic.xlsx and produces a ParsedData dataclass
containing a merged DataFrame and grading increments dict.

"""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# dimension field names (used throughout the project)
DIMENSION_FIELDS = [
    "stick_length",
    "lbp_length",
    "ball_girth",
    "waist_girth",
    "instep_girth",
    "lbp_ball_width",
    "lbp_heel_width",
    "heel_height",
    "toe_spring",
]

# file path
DEFAULT_XLSX = Path(__file__).parent.parent / "hilos_last_library_synthetic.xlsx"

# null-handling
def safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

# Column name normalisation
_LIBRARY_RENAME = {
    "Last Code": "last_code",
    "Vendor Code": "vendor_code",
    "Vendor": "vendor",
    "Gender": "gender",
    "Category": "category",
    "Width": "width",
    "Size (US)": "size_us",
    "Heel Template": "heel_template",
    "Toe Template": "toe_template",
    "Toe Angle (°)": "toe_angle",
    "Toe Thickness (mm)": "toe_thickness",
    "Date": "date",
    "Stick Length (mm)": "stick_length",
    "LBP Length (mm)": "lbp_length",
    "Ball Girth (mm)": "ball_girth",
    "Waist Girth (mm)": "waist_girth",
    "Instep Girth (mm)": "instep_girth",
    "LBP Ball Width (mm)": "lbp_ball_width",
    "LBP Heel Width (mm)": "lbp_heel_width",
    "Heel Height (mm)": "heel_height",
    "Toe Spring (mm)": "toe_spring",
}

_METADATA_RENAME = {
    "Last Code": "last_code",
    "Vendor Code": "vendor_code",
    "Vendor": "vendor",
    "Gender": "gender",
    "Category": "category",
    "Width": "width",
    "Heel Template": "heel_template",
    "Toe Template": "toe_template",
    "Toe Angle (°)": "toe_angle",
    "Toe Thickness (mm)": "toe_thickness",
    "Date Created": "date_created",
}


# ParsedData dataclass
@dataclass
class ParsedData:
    """Container returned by load_data().

    Attributes:
        lasts_df:  Merged DataFrame (Last Library + Metadata), one row per
                   (last_code, size_us) pair.  Numeric dimension columns are
                   float or None; never raw NaN.
        grading:   Dict mapping dimension field name → mm increment per full
                   US size (e.g. {"ball_girth": 4.0, ...}).
    """

    lasts_df: pd.DataFrame
    grading: dict[str, float]


# Main loader
def load_data(xlsx_path: str | Path = DEFAULT_XLSX) -> ParsedData:
    """Parse all three sheets and return a ParsedData instance.

    Args:
        xlsx_path: Path to the Excel file (read-only; never modified).

    Returns:
        ParsedData with merged DataFrame and grading dict.

    Raises:
        FileNotFoundError: If the xlsx file doesn't exist.
        ValueError: If required sheets are missing.
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # Sheet 1: Last Library
    lib_raw = pd.read_excel(path, sheet_name="Last Library", header=1)
    lib_df = lib_raw.rename(columns=_LIBRARY_RENAME)

    # Keep only rows with a valid last_code (drops any blank trailer rows)
    lib_df = lib_df[lib_df["last_code"].notna()].copy()

    # Normalise size_us to float
    lib_df["size_us"] = lib_df["size_us"].apply(safe_float)

    # Coerce all dimension columns through safe_float (eliminates NaN)
    for col in DIMENSION_FIELDS:
        if col in lib_df.columns:
            lib_df[col] = lib_df[col].apply(safe_float)

    # Deduplicate on (last_code, size_us); keep first occurrence
    before = len(lib_df)
    lib_df = lib_df.drop_duplicates(subset=["last_code", "size_us"], keep="first")
    after = len(lib_df)
    if before != after:
        import warnings
        warnings.warn(
            f"Dropped {before - after} duplicate (last_code, size_us) rows.",
            stacklevel=2,
        )

    # Sheet 2: Last Metadata
    meta_raw = pd.read_excel(path, sheet_name="Last Metadata", header=1)
    meta_df = meta_raw.rename(columns=_METADATA_RENAME)
    meta_df = meta_df[meta_df["last_code"].notna()].copy()

    # Columns that overlap with Last Library — drop from metadata before merge
    # to avoid _x/_y suffixes (keep Library versions as they are per-size)
    overlap = {"vendor_code", "vendor", "gender", "category", "width",
               "heel_template", "toe_template", "toe_angle", "toe_thickness"}
    meta_keep = ["last_code", "date_created"] + [
        c for c in meta_df.columns if c not in overlap and c != "last_code"
    ]
    # Simplest join: metadata supplies date_created; all other fields already
    # present in Last Library per-row.  We left-join to ensure no library rows
    # are lost even if metadata is incomplete.
    meta_slim = meta_df[["last_code", "date_created"]].drop_duplicates("last_code")
    merged = lib_df.merge(meta_slim, on="last_code", how="left")

    # Normalise gender to uppercase for consistent filtering
    if "gender" in merged.columns:
        merged["gender"] = merged["gender"].str.upper().str.strip()

    # Reset index for clean downstream use
    merged = merged.reset_index(drop=True)

    # Sheet 3: Grading Reference
    grading_raw = pd.read_excel(path, sheet_name="Grading Reference", header=1)
    grading_raw.columns = ["dimension", "field_name", "increment"]
    grading_raw = grading_raw[grading_raw["field_name"].notna()]

    grading: dict[str, float] = {}
    for _, row in grading_raw.iterrows():
        field = str(row["field_name"]).strip()
        inc = safe_float(row["increment"])
        if field and inc is not None:
            grading[field] = inc

    return ParsedData(lasts_df=merged, grading=grading)
