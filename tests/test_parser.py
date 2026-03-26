"""
Part 1: Unit tests for src/parser.py

Tests:
  1. test_all_sheets_load         — correct row counts for all 3 sheets
  2. test_metadata_join           — every row has vendor/gender populated after join
  3. test_missing_values_handled  — no raw NaN floats in output
  4. test_grading_reference_parsed — grading dict has all expected keys + numeric values
  5. test_no_silent_data_loss     — row count matches unique (last_code, size_us) pairs
  6. test_safe_float              — safe_float handles edge cases correctly
  7. test_dimension_columns_present — all canonical dimension fields present in DataFrame
"""

import math
from pathlib import Path

import pytest

from src.parser import DIMENSION_FIELDS, ParsedData, load_data, safe_float

# Path to the real Excel file
XLSX = Path(__file__).parent.parent / "hilos_last_library_synthetic.xlsx"


# safe_float unit tests (isolated, no file I/O)
class TestSafeFloat:
    def test_normal_float(self):
        assert safe_float(245.3) == 245.3

    def test_integer(self):
        assert safe_float(9) == 9.0

    def test_string_number(self):
        assert safe_float("12.5") == 12.5

    def test_none_returns_none(self):
        assert safe_float(None) is None

    def test_nan_returns_none(self):
        assert safe_float(float("nan")) is None

    def test_invalid_string_returns_none(self):
        assert safe_float("not_a_number") is None

    def test_empty_string_returns_none(self):
        assert safe_float("") is None


# Parser integration tests (require the xlsx file)
@pytest.fixture(scope="module")
def parsed() -> ParsedData:
    if not XLSX.exists():
        pytest.skip(f"Excel file not found: {XLSX}")
    return load_data(XLSX)


class TestAllSheetsLoad:
    """Test 1: All 3 sheets parse without error, correct row/grading counts."""

    def test_returns_parsed_data_instance(self, parsed):
        assert isinstance(parsed, ParsedData)

    def test_library_has_expected_rows(self, parsed):
        # 68–69 data rows (5 lasts × ~13–14 sizes each)
        assert 60 <= len(parsed.lasts_df) <= 75, (
            f"Expected ~68 rows, got {len(parsed.lasts_df)}"
        )

    def test_grading_has_entries(self, parsed):
        assert len(parsed.grading) >= 9, (
            f"Expected ≥9 grading entries, got {len(parsed.grading)}"
        )


class TestMetadataJoin:
    """Test 2: Every row in the merged DataFrame has required metadata fields."""

    def test_all_rows_have_last_code(self, parsed):
        null_codes = parsed.lasts_df["last_code"].isna().sum()
        assert null_codes == 0, f"{null_codes} rows missing last_code"

    def test_all_rows_have_size_us(self, parsed):
        null_sizes = parsed.lasts_df["size_us"].isna().sum()
        assert null_sizes == 0, f"{null_sizes} rows missing size_us"

    def test_known_last_codes_present(self, parsed):
        codes = set(parsed.lasts_df["last_code"].unique())
        expected = {"HS010125ML-1", "HS020325WL-1", "HS030624ML-2", "HS040924WL-2", "HS051124ML-3"}
        assert expected.issubset(codes), f"Missing codes: {expected - codes}"

    def test_gender_normalised_to_uppercase(self, parsed):
        genders = parsed.lasts_df["gender"].dropna().unique()
        for g in genders:
            assert g == g.upper(), f"Gender '{g}' is not uppercase"


class TestMissingValuesHandled:
    """Test 3: No raw NaN floats in dimension columns (coerced to None via safe_float)."""

    def test_no_nan_in_dimension_columns(self, parsed):
        df = parsed.lasts_df
        for col in DIMENSION_FIELDS:
            if col not in df.columns:
                continue
            for val in df[col]:
                if isinstance(val, float):
                    assert not math.isnan(val), (
                        f"Column '{col}' contains NaN — should have been coerced to None"
                    )


class TestGradingReferenceParsed:
    """Test 4: Grading dict has all expected dimension keys with numeric values."""

    EXPECTED_FIELDS = [
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

    def test_all_dimension_fields_present(self, parsed):
        for field in self.EXPECTED_FIELDS:
            assert field in parsed.grading, f"Missing grading key: '{field}'"

    def test_all_values_are_numeric(self, parsed):
        for field, val in parsed.grading.items():
            assert isinstance(val, (int, float)), (
                f"Grading value for '{field}' is not numeric: {val!r}"
            )
            assert not math.isnan(val), f"Grading value for '{field}' is NaN"

    def test_known_grading_values(self, parsed):
        # Spot-check known values from the source data
        assert parsed.grading["stick_length"] == pytest.approx(8.47, abs=0.01)
        assert parsed.grading["ball_girth"] == pytest.approx(4.0, abs=0.01)
        assert parsed.grading["heel_height"] == pytest.approx(0.0, abs=0.01)


class TestNoSilentDataLoss:
    """Test 5: Row count matches unique (last_code, size_us) pairs — no silent deduplication."""

    def test_row_count_equals_unique_pairs(self, parsed):
        df = parsed.lasts_df
        unique_pairs = df.drop_duplicates(subset=["last_code", "size_us"])
        assert len(df) == len(unique_pairs), (
            f"DataFrame has {len(df)} rows but only {len(unique_pairs)} unique "
            f"(last_code, size_us) pairs — unexpected duplicates present"
        )

    def test_five_lasts_present(self, parsed):
        n_lasts = parsed.lasts_df["last_code"].nunique()
        assert n_lasts == 5, f"Expected 5 unique lasts, got {n_lasts}"


class TestDimensionColumnsPresent:
    """Test 7: All canonical dimension fields exist in the merged DataFrame."""

    def test_all_dimension_fields_in_dataframe(self, parsed):
        missing = [f for f in DIMENSION_FIELDS if f not in parsed.lasts_df.columns]
        assert not missing, f"Missing dimension columns: {missing}"

    def test_spot_check_known_value(self, parsed):
        # HS010125ML-1 size 9 ball_girth should be 243.8
        df = parsed.lasts_df
        row = df[(df["last_code"] == "HS010125ML-1") & (df["size_us"] == 9.0)]
        assert len(row) == 1, "Expected exactly one row for HS010125ML-1 size 9"
        assert row.iloc[0]["ball_girth"] == pytest.approx(243.8, abs=0.1)


class TestEvalGroundTruth:
    """Test 8: Spot-checks that verify ground-truth values used in eval/test_cases.json.

    These act as a contract between the raw xlsx data and the evaluation harness —
    if a value changes in the source file the test catches it before the evaluator
    reports a false failure.
    """

    # --- TC1 / TC11 / TC12 ---
    def test_tc1_ball_girth_hs010125_size9(self, parsed):
        df = parsed.lasts_df
        val = df[(df.last_code == "HS010125ML-1") & (df.size_us == 9.0)].iloc[0]["ball_girth"]
        assert val == pytest.approx(243.8, abs=0.05)

    # --- TC2 ---
    def test_tc2_heel_height_hs010125_size10_5(self, parsed):
        df = parsed.lasts_df
        val = df[(df.last_code == "HS010125ML-1") & (df.size_us == 10.5)].iloc[0]["heel_height"]
        assert val == pytest.approx(12.1, abs=0.05)

    # --- TC3 ---
    def test_tc3_toe_spring_hs030624_size8(self, parsed):
        df = parsed.lasts_df
        val = df[(df.last_code == "HS030624ML-2") & (df.size_us == 8.0)].iloc[0]["toe_spring"]
        assert val == pytest.approx(19.4, abs=0.05)

    # --- TC4: max lbp_ball_width @ size 10 → HS030624ML-2 ---
    def test_tc4_max_lbp_ball_width_size10(self, parsed):
        df = parsed.lasts_df
        sub = df[df.size_us == 10.0][["last_code", "lbp_ball_width"]].dropna()
        winner = sub.loc[sub["lbp_ball_width"].idxmax(), "last_code"]
        assert winner == "HS030624ML-2"

    # --- TC5: min stick_length @ size 9 → HS020325WL-1 ---
    def test_tc5_min_stick_length_size9(self, parsed):
        df = parsed.lasts_df
        sub = df[df.size_us == 9.0][["last_code", "stick_length"]].dropna()
        winner = sub.loc[sub["stick_length"].idxmin(), "last_code"]
        assert winner == "HS020325WL-1"
        # Verify the actual minimum value
        assert sub["stick_length"].min() == pytest.approx(266.7, abs=0.1)

    # --- TC6: mens stick_length > 290 @ size 11 → 3 lasts ---
    def test_tc6_mens_stick_over_290_size11(self, parsed):
        df = parsed.lasts_df
        sub = df[(df.size_us == 11.0) & (df.gender == "MENS")]
        qualifying = set(sub[sub.stick_length > 290]["last_code"].tolist())
        assert qualifying == {"HS010125ML-1", "HS030624ML-2", "HS051124ML-3"}

    # --- TC7: ball_girth < 230 @ size 7 → 2 lasts ---
    def test_tc7_ball_girth_under_230_size7(self, parsed):
        df = parsed.lasts_df
        sub = df[df.size_us == 7.0]
        qualifying = set(sub[sub.ball_girth < 230]["last_code"].tolist())
        assert qualifying == {"HS020325WL-1", "HS040924WL-2"}
        # Verify the exact values
        val_wl1 = sub[sub.last_code == "HS020325WL-1"].iloc[0]["ball_girth"]
        val_wl2 = sub[sub.last_code == "HS040924WL-2"].iloc[0]["ball_girth"]
        assert val_wl1 == pytest.approx(223.9, abs=0.05)
        assert val_wl2 == pytest.approx(228.8, abs=0.05)

    # --- TC8: grading estimate HS010125ML-1 size 9→8 ball_girth ---
    def test_tc8_grading_estimate_ball_girth(self, parsed):
        df = parsed.lasts_df
        known = df[(df.last_code == "HS010125ML-1") & (df.size_us == 9.0)].iloc[0]["ball_girth"]
        rate = parsed.grading["ball_girth"]
        estimated = known + (8 - 9) * rate
        assert estimated == pytest.approx(239.8, abs=0.05)
        # Also assert actual size-8 value matches
        actual = df[(df.last_code == "HS010125ML-1") & (df.size_us == 8.0)].iloc[0]["ball_girth"]
        assert actual == pytest.approx(239.8, abs=0.05)

    # --- TC9: grading estimate HS030624ML-2 size 9→10 waist_girth ---
    def test_tc9_grading_estimate_waist_girth(self, parsed):
        df = parsed.lasts_df
        known = df[(df.last_code == "HS030624ML-2") & (df.size_us == 9.0)].iloc[0]["waist_girth"]
        rate = parsed.grading["waist_girth"]
        estimated = known + (10 - 9) * rate
        assert estimated == pytest.approx(248.9, abs=0.05)
        actual = df[(df.last_code == "HS030624ML-2") & (df.size_us == 10.0)].iloc[0]["waist_girth"]
        assert actual == pytest.approx(248.9, abs=0.05)

    # --- TC10: max instep_girth womens @ size 7 → 227.0 ---
    def test_tc10_max_instep_girth_womens_size7(self, parsed):
        df = parsed.lasts_df
        sub = df[(df.size_us == 7.0) & (df.gender == "WOMENS")]
        assert sub["instep_girth"].max() == pytest.approx(227.0, abs=0.05)
        winner = sub.loc[sub["instep_girth"].idxmax(), "last_code"]
        assert winner == "HS040924WL-2"
