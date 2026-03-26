from __future__ import annotations

import json
import os

import streamlit as st

# Page config
st.set_page_config(
    page_title="HILOS Last Library",
    page_icon="👟",
    layout="centered",
)

# HILOS brand theme — black bg, orange accent, grey tones
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0D0D0D;
        color: #E8E8E8;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }
    [data-testid="stHeader"] { background-color: #0D0D0D; }
    [data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid #2A2A2A;
    }

    /* ── Typography ── */
    h1, h2, h3, h4 { color: #FFFFFF; letter-spacing: -0.02em; }
    p, li, span, label { color: #C8C8C8; }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #1A1A1A;
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInputContainer"] textarea {
        color: #E8E8E8 !important;
        background-color: transparent !important;
    }
    [data-testid="stChatInputContainer"] button {
        background-color: #E8572A !important;
        border-radius: 6px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #1E1E1E;
        color: #E8E8E8;
        border: 1px solid #333333;
        border-radius: 6px;
        font-size: 13px;
        transition: background 0.15s, border-color 0.15s;
    }
    .stButton > button:hover {
        background-color: #E8572A;
        border-color: #E8572A;
        color: #FFFFFF;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background-color: #1A1A1A;
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #E8572A !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #888888 !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        overflow: hidden;
    }
    .dvn-scroller { background-color: #161616 !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #161616;
        border: 1px solid #2A2A2A;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary { color: #888888; font-size: 12px; }

    /* ── Info / Warning / Error ── */
    [data-testid="stAlert"] { border-radius: 6px; }
    .stInfo  { background-color: #1A1A1A !important; border-left: 3px solid #E8572A !important; }
    .stWarning { background-color: #1A1A1A !important; border-left: 3px solid #F5A623 !important; }
    .stError   { background-color: #1A1A1A !important; border-left: 3px solid #E8572A !important; }

    /* ── Divider ── */
    hr { border-color: #2A2A2A; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1A1A1A; }
    ::-webkit-scrollbar-thumb { background: #333333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #E8572A; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<h1 style="font-size:2rem;font-weight:700;letter-spacing:-0.03em;">'
    '<span style="color:#E8572A;">HILOS</span> Last Library</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "Ask questions about shoe last dimensions in natural language. "
)

# Load pipeline
@st.cache_resource
def get_pipeline():
    from dotenv import load_dotenv
    load_dotenv()
    from src.pipeline import Pipeline
    return Pipeline()


# Check API key before loading
if not os.environ.get("ANTHROPIC_API_KEY"):
    # Try loading from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY not set. "
        "Create a `.env` file with `ANTHROPIC_API_KEY=sk-ant-...` and restart."
    )
    st.stop()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Session state — chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_result"):
            with st.expander("Source data", expanded=False):
                st.json(msg["tool_result"])

# Suggested queries (shown only when chat is empty)
if not st.session_state.messages:
    st.markdown(
        '<p style="color:#888888;font-size:13px;margin-bottom:8px;">Try asking:</p>',
        unsafe_allow_html=True,
    )
    examples = [
        "What is the ball girth of HS010125ML-1 in size 9?",
        "Which last has the widest ball width in size 10?",
        "List all men's lasts with a stick length over 290mm in size 11.",
        "If I need size 8 but only have size 9 data for HS010125ML-1, what would the estimated ball girth be?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"example_{i}", use_container_width=True):
            st.session_state["prefill"] = ex
            st.rerun()

# Handle prefilled example
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask about last dimensions…") or prefill

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Querying dataset..."):
            result = pipeline.ask(user_input)

        answer = result.get("answer", "")
        query_type = result.get("query_type") or ""
        tool_result = result.get("tool_result")
        answer_value = result.get("answer_value")

        # ---- Render answer based on query_type ----
        if query_type == "lookup" and isinstance(answer_value, (int, float)):
            # Metric card
            label = result.get("tool_called") or "Value"
            st.metric(label=label, value=f"{answer_value} mm")
            st.markdown(answer)

        elif query_type in ("comparison", "filter") and isinstance(tool_result, dict):
            # Table view
            rows = tool_result.get("results", [])
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            st.markdown(answer)

        elif query_type == "grading" and isinstance(answer_value, (int, float)):
            # Metric card with estimate caveat
            st.metric(label="Estimated value", value=f"{answer_value} mm")
            st.markdown(answer)
            if tool_result and tool_result.get("grading_note"):
                st.info(f"Formula: {tool_result['grading_note']}")

        elif query_type == "ambiguous":
            st.warning(answer)

        else:
            st.markdown(answer)

        # Expandable source data
        if tool_result:
            with st.expander("Source data", expanded=False):
                st.json(tool_result)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_result": tool_result,
    })

# Sidebar — dataset info
with st.sidebar:
    st.markdown(
        '<p style="color:#E8572A;font-weight:700;font-size:11px;'
        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">Dataset</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Last Library")
    try:
        data = pipeline.data
        df = data.lasts_df
        st.metric("Total rows", len(df))
        st.metric("Unique lasts", df["last_code"].nunique())
        st.subheader("Available lasts")
        for code in sorted(df["last_code"].dropna().unique()):
            row = df[df["last_code"] == code].iloc[0]
            sizes = sorted(df[df["last_code"] == code]["size_us"].dropna().tolist())
            gender = row.get("gender", "")
            st.markdown(f"**{code}** ({gender})  \nSizes: {sizes[0]}–{sizes[-1]}")
    except Exception:
        st.warning("Could not load dataset info.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
