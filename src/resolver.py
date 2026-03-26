"""
Ambiguity handling and input normalization.

Runs deterministically BEFORE any LLM call.  If the input is ambiguous
(e.g. partial last code matching multiple lasts), this module raises
AmbiguityError and the pipeline returns the clarification prompt to the
user directly — the LLM is never reached.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional


DIMENSION_SYNONYMS: dict[str, str] = {
    # stick_length
    "stick length": "stick_length",
    "overall length": "stick_length",
    "last length": "stick_length",
    "length": "stick_length",
    # lbp_length
    "lbp length": "lbp_length",
    "bottom pattern length": "lbp_length",
    "last bottom pattern length": "lbp_length",
    # ball_girth
    "ball girth": "ball_girth",
    "forefoot girth": "ball_girth",
    "forefoot circumference": "ball_girth",
    "ball circumference": "ball_girth",
    # waist_girth
    "waist girth": "waist_girth",
    "waist circumference": "waist_girth",
    # instep_girth
    "instep girth": "instep_girth",
    "instep circumference": "instep_girth",
    # lbp_ball_width
    "lbp ball width": "lbp_ball_width",
    "ball width": "lbp_ball_width",
    "forefoot width": "lbp_ball_width",
    "width at ball": "lbp_ball_width",
    "bottom pattern ball width": "lbp_ball_width",
    # lbp_heel_width
    "lbp heel width": "lbp_heel_width",
    "heel width": "lbp_heel_width",
    "bottom pattern heel width": "lbp_heel_width",
    # heel_height
    "heel height": "heel_height",
    "heel elevation": "heel_height",
    # toe_spring
    "toe spring": "toe_spring",
    "toe angle": "toe_spring",
}

# Gender normalization map
GENDER_SYNONYMS: dict[str, str] = {
    "men": "MENS",
    "mens": "MENS",
    "men's": "MENS",
    "male": "MENS",
    "m": "MENS",
    "women": "WOMENS",
    "womens": "WOMENS",
    "women's": "WOMENS",
    "female": "WOMENS",
    "w": "WOMENS",
    "f": "WOMENS",
}


# Exceptions
class AmbiguityError(Exception):
    """Raised when a partial last code matches more than one last.

    Attributes:
        partial: The partial code the user provided.
        candidates: List of matching last codes.
    """

    def __init__(self, partial: str, candidates: list[str]):
        self.partial = partial
        self.candidates = candidates
        opts = ", ".join(candidates)
        super().__init__(
            f"Ambiguous last code '{partial}' — did you mean one of: {opts}?"
        )

    def clarification_message(self) -> str:
        opts = " or ".join(f"'{c}'" for c in self.candidates)
        return f"I found multiple lasts matching '{self.partial}': {opts}. Which one did you mean?"


# ResolvedContext — output of pre_process()
@dataclass
class ResolvedContext:
    """Normalised values extracted from the raw query context."""

    last_code: Optional[str] = None          # Exact code after fuzzy resolution
    size_us: Optional[float] = None
    dimension: Optional[str] = None          # Canonical field name
    gender: Optional[str] = None             # "MENS" or "WOMENS"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    # Carry-through of anything else the caller passed
    extras: dict = field(default_factory=dict)


# Core resolver
class Resolver:
    """Stateful resolver that holds the set of known last codes."""

    def __init__(self, known_last_codes: list[str]):
        self._known = [c.upper() for c in known_last_codes]
        self._original = {c.upper(): c for c in known_last_codes}

    # Public API
    def resolve_last_code(self, raw: str) -> str:
        """Return the canonical last code for *raw*, or raise AmbiguityError.

        Matching is case-insensitive.  Exact match wins immediately.
        If no exact match, fuzzy matching is attempted (cutoff=0.6).
        - Exactly 1 fuzzy match → return it.
        - More than 1 fuzzy match → raise AmbiguityError.
        - Zero matches → raise ValueError with available codes.
        """
        upper = raw.strip().upper()

        # Exact match (case-insensitive)
        if upper in self._known:
            return self._original[upper]

        # Fuzzy match
        matches = get_close_matches(upper, self._known, n=5, cutoff=0.6)
        if len(matches) == 1:
            return self._original[matches[0]]
        if len(matches) > 1:
            originals = [self._original[m] for m in matches]
            raise AmbiguityError(raw, originals)

        # Substring match as fallback
        sub_matches = [k for k in self._known if upper in k or k in upper]
        if len(sub_matches) == 1:
            return self._original[sub_matches[0]]
        if len(sub_matches) > 1:
            originals = [self._original[m] for m in sub_matches]
            raise AmbiguityError(raw, originals)

        available = ", ".join(sorted(self._original.values()))
        raise ValueError(
            f"Unknown last code '{raw}'. Available codes: {available}"
        )

    def resolve_dimension(self, raw: str) -> str:
        """Return canonical field name for a dimension string.

        Accepts exact field names (e.g. 'ball_girth') or natural language
        synonyms (e.g. 'forefoot girth').  Raises ValueError if unrecognised.
        """
        from src.parser import DIMENSION_FIELDS

        normalised = raw.strip().lower().replace("_", " ")

        # Direct field name match
        if raw.strip().lower() in DIMENSION_FIELDS:
            return raw.strip().lower()

        # Synonym map
        if normalised in DIMENSION_SYNONYMS:
            return DIMENSION_SYNONYMS[normalised]

        # Fuzzy match against synonym keys + field names
        candidates = list(DIMENSION_SYNONYMS.keys()) + DIMENSION_FIELDS
        matches = get_close_matches(normalised, candidates, n=3, cutoff=0.7)
        if matches:
            best = matches[0]
            if best in DIMENSION_SYNONYMS:
                return DIMENSION_SYNONYMS[best]
            return best

        raise ValueError(
            f"Unrecognised dimension '{raw}'. "
            f"Known fields: {', '.join(DIMENSION_FIELDS)}"
        )

    @staticmethod
    def resolve_gender(raw: str) -> str:
        """Normalise gender string to 'MENS' or 'WOMENS'. Raises ValueError if unrecognised."""
        key = raw.strip().lower().rstrip("'s")
        # Try direct lookup
        if raw.strip().lower() in GENDER_SYNONYMS:
            return GENDER_SYNONYMS[raw.strip().lower()]
        if key in GENDER_SYNONYMS:
            return GENDER_SYNONYMS[key]
        # Allow already-normalised values
        upper = raw.strip().upper()
        if upper in ("MENS", "WOMENS"):
            return upper
        raise ValueError(
            f"Unrecognised gender '{raw}'. Use 'mens'/'men' or 'womens'/'women'."
        )

    def known_codes(self) -> list[str]:
        return sorted(self._original.values())
