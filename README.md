# HILOS Last Library — LLM Query System

A natural language interface for querying HILOS Studio's synthetic shoe-last specification dataset. Built for the AI Research Fellowship test.

The UI follows HILOS Lab's website color tone — mostly black and orange — to echo the brand's logo and visual identity.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run unit tests (Part 1)

```bash
pytest tests/ -v
```

### 4. Run the pipeline CLI (Part 2)

```bash
python -m src.pipeline
# or with a custom query:
python -m src.pipeline "What is the ball girth of HS010125ML-1 in size 9?"
```

### 5. Run the evaluation harness (Part 3)

```bash
python -m src.evaluator
```

### 6. Launch the Streamlit chat UI (Part 4)

```bash
streamlit run app.py
```

---

## Project Structure

```
HilosProject/
├── hilos_last_library_synthetic.xlsx   ← read-only source data
├── src/
│   ├── parser.py       # Data loading + safe_float()
│   ├── resolver.py     # Ambiguity handling
│   ├── query_tools.py  # 5 deterministic query functions + tool schemas
│   ├── pipeline.py     # LLM orchestration + format_response + log_trace
│   └── evaluator.py    # Evaluation harness
├── tests/
│   └── test_parser.py  # Unit tests for Part 1
├── eval/
│   └── test_cases.json # 12 ground-truth Q&A pairs
├── logs/               # Query traces (gitignored)
├── app.py              # Streamlit chat UI
├── requirements.txt
└── .env.example
```

---

## Architecture

### Query flow

```
user_query
  → resolver.pre_process()    # ambiguity blocking, code normalization
  → Claude (tool schemas)     # decides which tool to call
  → Python tool execution     # real DataFrame query
  → Claude (tool result)      # synthesizes natural language answer
  → format_response()         # presentation shaping
  → log_trace()               # JSON Lines trace
  → structured output dict
```

### Five tools

| Tool | Purpose |
|------|---------|
| `lookup_dimension` | Exact value for a last + size + dimension |
| `compare_lasts` | Rank all lasts at a size by a dimension |
| `filter_lasts` | Filter by gender, size, dimension range |
| `estimate_graded` | Linear grading extrapolation to a new size |
| `list_lasts` | Enumerate available lasts (meta, rarely needed) |

---

## Part 1 — Data Parsing & Normalization

### How I parse all three sheets

I use `pandas.read_excel()` with `header=1` to skip the decorative title row that sits above the real headers. Column names get normalized to `snake_case` — stripping units like "(mm)" and "(°)" and replacing spaces with underscores. Last Library and Last Metadata are joined on `last_code` via a left merge, and Grading Reference is parsed into a plain Python dict keyed by field name.

### How I handle missing values

Everything numeric goes through `safe_float(val)` — a single utility function I defined in `parser.py` and imported everywhere else. It converts `NaN`, `None`, and malformed strings to `None` consistently. The reason to use `None` over `NaN` is that `None` is JSON-serializable and `NaN` is not, which matters when the pipeline returns structured output dicts.

### Edge cases

The main ones I ran into: the title row at row 0 tricks pandas into reading garbage column names (fixed by `header=1`), sizes are stored as floats like `9.0` rather than integers (normalized via `safe_float()`), and there are potential duplicate `(last_code, size_us)` rows that get dropped with `drop_duplicates` after the join.

### Why DataFrame and not JSON

DataFrame lets me use pandas vectorized operations directly in the query tools — `nlargest`, `nsmallest`, boolean mask chaining for `filter_lasts`, `groupby` for comparisons. Storing as JSON would mean reimplementing all of that manually on every query. The grading dict is the one exception — it's a simple key-value lookup so a plain dict makes more sense there than a DataFrame.

### Unit tests

Five tests in `tests/test_parser.py`:

- `test_all_sheets_load` — all 3 sheets parse without error, row counts match expected
- `test_metadata_join` — every row in the merged DataFrame has `vendor` and `gender` populated
- `test_missing_values_handled` — no raw `NaN` floats in the output, only `None`
- `test_grading_reference_parsed` — grading dict has all 9 expected keys with numeric values
- `test_no_silent_data_loss` — row count after dedup matches unique `(last_code, size_us)` pairs

---

## Part 2 — LLM-Accessible Knowledge Layer

### Why tool use, not RAG or context injection

The dataset is 68 rows. Embedding and vector retrieval adds cost and latency for no real benefit at this size, and semantic similarity search is fundamentally unreliable for exact numeric values — "290mm stick length" is not a text similarity problem. RAG also risks blending nearby rows and producing hallucinated composite answers.

Plain context injection has the opposite problem: pushing all 68 rows × 9 dimensions into every prompt is wasteful and noisy, and you can't do grading arithmetic inline without code execution anyway.

Tool use is the right call here because it guarantees that every number in the answer comes from a real DataFrame query. The LLM decides *which* tool to call; Python executes it; the LLM synthesizes from the actual result.

### API and model

I used the Anthropic API with `claude-sonnet-4-6` via the `anthropic` Python SDK, with the key loaded from `.env`.

### What goes into the prompt

Rather than injecting data into the prompt, I only send the tool schemas and the normalized query. Claude picks the tool, Python runs it, and only the specific result comes back into context. This keeps the prompt minimal and precisely relevant — no padding with data the LLM doesn't need for this particular query.

The system prompt carries the anti-hallucination contract:

> "You have no knowledge of this dataset. You MUST call a tool before stating any numerical fact. Do not guess or infer values from memory. If no tool result is available, say 'I don't have that data' — never fabricate a number."

It also instructs Claude to prefer `lookup_dimension` over `estimate_graded` when the requested size already exists in the data.

### How I handle missing data

Every tool returns a structured result object rather than raw `None`. If a size doesn't exist, the response includes `"available_sizes": [...]`. If the last code is unknown, it includes `"closest_matches": [...]`. Claude then synthesizes these into a helpful natural language response — listing alternatives rather than guessing or returning a blank answer.

### Structured output

Every pipeline response returns:

```json
{
  "answer": "The ball girth is 243.8 mm.",
  "answer_value": 243.8,
  "query_type": "lookup",
  "tool_called": "lookup_dimension",
  "tool_result": {...},
  "error": null
}
```

`answer_value` is always a primitive — float, list, or null — so the evaluator reads it directly without fragile string parsing.

### Ambiguity handling

`resolver.py` runs before any LLM call. Partial last codes get fuzzy-matched against known codes using `difflib`; if more than one candidate matches, an `AmbiguityError` is raised and returned to the user as a clarification prompt — the LLM is never called, no API cost, no hallucination risk. Informal dimension names map to canonical field names via a synonym dict, and gender terms like "mens" normalize to "Men".

---

## Part 3 — Evaluation

### Why these 12 test cases

I wanted coverage across all four query types rather than stacking easy lookups. The cases also include two structural tests that check *how* the pipeline behaves, not just what it outputs — test case 11 verifies the LLM calls `lookup_dimension` and not `estimate_graded` when the size already exists in the data, and test case 12 checks that `list_lasts` is never called when the last code is already explicit. Ground truth for every case was verified directly from the raw Excel file.

| Type | Count | Description |
|------|-------|-------------|
| `lookup` | 3 | Direct dimension lookup |
| `comparison` | 2 | Which last has max/min |
| `filter` | 3 | Range filter queries |
| `grading` | 2 | Linear grading estimates |
| `tool_selection` | 1 | Correct tool chosen when size exists |
| `no_list_lasts` | 1 | `list_lasts` not called unnecessarily |

### Pass criteria

- **Numeric answers** — pass if `abs(answer_value - expected) <= 1.0` (±1mm tolerance)
- **List answers** — pass if `set(answer_value) == set(expected)` (set equality, order doesn't matter)
- **Tool selection** — pass if logged `tool_called` matches `expected_tool`

### Accuracy

Target is ≥80%, reported as a per-query-type breakdown. The evaluator reads `answer_value` directly from the pipeline's structured output — no natural language parsing involved.

### Failure modes observed

**1. Grading logic hallucination**
The LLM occasionally computes a grading estimate from memory rather than calling `estimate_graded` — particularly for simple round-number sizes where the math feels "obvious." Test case 11 specifically detects this by checking `tool_called` in the trace log. Fix: set temperature to 0 and use stricter tool-use enforcement in the system prompt. The trace log makes this diagnosable — if `tool_called` is missing or wrong, it's a system prompt problem; if the tool was called correctly but the answer is wrong, it's a query_tools logic problem.

**2. Size format mismatch**
Users type "size 9" but the data stores `9.0`. The resolver handles most cases via `safe_float()` coercion, but inputs like "nine" or "size nine" are not covered. Fix: extend the resolver with a number-word normalization map before the `safe_float()` call.

**3. Over-eager fuzzy matching**
If a partial code like "HS010" happens to score highly against only one last in `difflib`, it resolves silently rather than asking for confirmation. Fix: raise the minimum confidence threshold and always require exact match or explicit user confirmation — never silently resolve a code that wasn't typed exactly.

---

## Part 4 — Streamlit Chat UI

I chose to build the UI first because it's the most direct way to actually *use* the pipeline — typing queries and seeing responses makes edge cases obvious in a way that a CLI does not. The UI follows HILOS Lab's website color palette, using mostly black and orange to echo the brand's logo.

- Chat interface via `st.chat_message` / `st.chat_input`
- Lookup and grading results → `st.metric` cards
- Comparison and filter results → `st.dataframe`
- Expandable "Source data" expander showing the raw tool result JSON — useful for verifying that answers came from real data, not synthesis
- Sidebar with dataset overview and available lasts

---

## What I'd Do Differently With More Time

**Cross-last similarity queries (stretch B)**
For each last, I'd compute a measurement vector from its size-10 row across all 9 dimensions, normalize to z-scores, and use cosine similarity to rank all other lasts against the query last's vector. A gender filter would apply before ranking. This would be exposed as a new `find_similar_last` tool — the LLM calls it the same way it calls any other tool, so the pipeline architecture stays the same.

**Confidence and uncertainty signals (stretch C)**
For grading estimates, confidence is deterministic: flag any extrapolation spanning more than 2 sizes as uncertain and surface the interval `±(n_sizes × grading_rate)` alongside the answer. For lookup and filter queries, uncertainty comes from ambiguous input rather than data gaps — the existing `AmbiguityError` mechanism already handles the hard block case, but I'd extend it to surface low-confidence fuzzy matches as a soft warning ("I found a possible match — did you mean X?") rather than either blocking or silently proceeding.

**Other things on the list**
- Streaming Claude's response tokens to the UI rather than waiting for the full response
- Caching tool results to avoid redundant DataFrame queries on repeated questions
- Using an LLM judge to evaluate answer quality beyond exact numeric match — phrasing, caveats, completeness
- Extracting `format_response()` and `log_trace()` into separate `formatter.py` and `logger.py` modules once the pipeline is fully verified