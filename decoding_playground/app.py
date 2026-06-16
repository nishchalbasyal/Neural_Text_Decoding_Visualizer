# app.py - Streamlit UI for the decoding playground. Model logic lives in decoder.py.

import streamlit as st
import plotly.graph_objects as go

from decoder import load_model, repetition_score, step_once

st.set_page_config(page_title="Neural Decoding Playground", layout="wide")


# loads GPT-2 once per server process, cached across reruns
@st.cache_resource(show_spinner="Loading GPT-2 model…  (this takes a few seconds on first run)")
def _ensure_model_loaded() -> bool:
    load_model()
    return True


_ensure_model_loaded()

DEFAULT_PROMPT = (
    "The unicorn lived in a magical forest, and it had a beautiful silver horn. "
    "One day, a young girl named Lily discovered the unicorn"
)

STRATEGY_LABELS = {
    "greedy": "Greedy",
    "beam": "Beam Search",
    "topk": "Top-k Sampling",
    "nucleus": "Nucleus (Top-p)",
}

HOW_IT_WORKS_MD = """
### How each strategy works

**Greedy** — always picks the top token. Fast, but repeats itself a lot.

**Beam Search** — tracks the best few sequences at once. Better than greedy, still repeats on long text.

**Top-k** — only samples from the k best tokens. Simple, but k doesn't adapt to the model's confidence.

**Nucleus (Top-p)** — keeps the smallest set of tokens that add up to probability p. Shrinks when the model is sure, grows when it's not.
"""


# GREEN = kept, RED = cut, GOLD = sampled
def build_chart(step_data: dict) -> go.Figure:
    if not step_data or "candidates" not in step_data:
        return go.Figure()

    candidates = step_data["candidates"]
    tokens = [c["token"].replace(" ", "·") for c in candidates]
    probs = [c["prob"] for c in candidates]

    colors = []
    for c in candidates:
        if c["chosen"]:
            colors.append("#f0c040")
        elif c["kept"]:
            colors.append("#3cb371")
        else:
            colors.append("#dc143c")

    fig = go.Figure(
        go.Bar(
            x=tokens,
            y=probs,
            marker_color=colors,
            text=[f"{prob:.3f}" for prob in probs],
            textposition="outside",
            textfont=dict(size=10),
        )
    )

    max_y = max(probs) if probs else 1.0
    fig.update_layout(
        title=dict(
            text=step_data.get("explanation", ""),
            font=dict(size=12),
            x=0,
        ),
        xaxis_title="Token (· = space)",
        yaxis_title="Probability (raw, before filtering)",
        yaxis=dict(range=[0, max_y * 1.30]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=75, l=50, r=20, b=70),
        showlegend=False,
        height=400,
        font=dict(family="monospace", color="#222222"),
    )
    fig.update_yaxes(
        gridcolor="#e8e8e8",
        zerolinecolor="#cccccc",
        showgrid=True,
    )
    fig.update_xaxes(showgrid=False)
    return fig


# PEAKED/FLAT badge, orange for peaked, blue for flat
def dist_html(label: str, metric: float) -> str:
    if label == "PEAKED":
        color = "#c45e00"
        bg = "#fff3e0"
        note = (
            "Model is confident — fixed-k risks over-truncating when the "
            "distribution is wider; nucleus shrinks to just the top tokens."
        )
    else:
        color = "#005fa3"
        bg = "#e3f2fd"
        note = (
            "Model is uncertain — fixed-k may cut many good tokens; "
            "nucleus expands to cover the necessary probability mass."
        )
    return (
        f'<div style="padding:10px 14px;border-radius:8px;'
        f'background:{bg};border:2px solid {color};margin:4px 0">'
        f'<span style="color:{color};font-size:1.15em;font-weight:bold">'
        f'{label}</span>'
        f'<span style="color:#444;font-size:0.88em;margin-left:10px">'
        f'top-token p={metric:.3f} — {note}</span></div>'
    )


# bundles slider values into the params dict decoder.py expects
def collect_params(temperature, k, p, beam_width) -> dict:
    return {
        "temperature": float(temperature),
        "k": int(k),
        "p": float(p),
        "beam_width": int(beam_width),
    }


# session state
if "steps" not in st.session_state:
    st.session_state.steps = []
if "current_text" not in st.session_state:
    st.session_state.current_text = ""

st.markdown(
    "# Neural Text Decoding Playground\n"
    "Reproduce the core ideas of **Holtzman et al. 2020 — *The Curious Case "
    "of Neural Text Degeneration*** using GPT-2 (124 M parameters, CPU).\n\n"
    "Generate one token at a time and see **exactly why** each token was chosen: "
    "which candidates existed, which were kept or cut, and which was finally sampled."
)

left, right = st.columns([1, 2])

# left column - controls
with left:

    prompt_box = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        height=120,
        placeholder="Type a prompt here…",
    )

    strategy = st.selectbox(
        "Decoding Strategy",
        options=list(STRATEGY_LABELS.keys()),
        format_func=lambda key: STRATEGY_LABELS[key],
        index=list(STRATEGY_LABELS.keys()).index("nucleus"),
        help="Select one of the six strategies from the paper.",
    )

    # only show the slider for the chosen strategy
    temperature, k, p, beam_width = 1.0, 50, 0.9, 4
    if strategy == "topk":
        k = st.slider(
            "Top-k (k)", min_value=1, max_value=500, value=50, step=1,
            help="Number of tokens to keep", key="k_slider",
        )
    elif strategy == "nucleus":
        p = st.slider(
            "Nucleus p", min_value=0.05, max_value=1.0, value=0.9, step=0.01,
            help="Cumulative probability threshold", key="p_slider",
        )
    elif strategy == "beam":
        beam_width = st.slider(
            "Beam Width (b)", min_value=2, max_value=10, value=4, step=1,
            help="Number of parallel sequences to track", key="beam_slider",
        )

    seed = st.number_input(
        "Random seed",
        value=42, step=1, format="%d",
        help="Fix the seed for reproducible sampling",
    )

    btn_col1, btn_col2 = st.columns(2)
    step_clicked = btn_col1.button("Step Once", type="primary", use_container_width=True)
    reset_clicked = btn_col2.button("Reset", use_container_width=True)

    if step_clicked:
        context = st.session_state.current_text or prompt_box.strip() or DEFAULT_PROMPT
        params = collect_params(temperature, k, p, beam_width)
        step_seed = int(seed) + len(st.session_state.steps)

        token_str, step_data = step_once(context, strategy, params, seed=step_seed)

        st.session_state.current_text = context + token_str
        st.session_state.steps.append(step_data)

    if reset_clicked:
        st.session_state.current_text = ""
        st.session_state.steps = []

    with st.expander("How each strategy works"):
        st.markdown(HOW_IT_WORKS_MD)

# right column - outputs
with right:

    steps = st.session_state.steps
    display_step = steps[-1] if steps else None

    display_text = st.session_state.current_text or prompt_box

    st.text_area(
        "Generated text  (most recent token appended at the end)",
        value=display_text,
        height=140,
        disabled=True,
    )

    if display_step:
        st.markdown(
            dist_html(display_step["dist_label"], display_step["dist_metric"]),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<i>Run a step to see the distribution shape badge.</i>", unsafe_allow_html=True)

    st.text_area(
        "Why this token? (plain-English explanation)",
        value=display_step.get("explanation", "") if display_step else "",
        height=70,
        disabled=True,
    )

    fig = build_chart(display_step) if display_step else go.Figure()
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        theme=None,
    )
    st.caption("Next-token probability distribution  [GREEN = kept  |  RED = cut  |  GOLD = sampled]")

    score_col1, score_col2 = st.columns(2)
    rep_display = f"{repetition_score(st.session_state.current_text) * 100:.1f}%" if st.session_state.current_text else ""
    score_col1.text_input(
        "Repetition score",
        value=rep_display,
        disabled=True,
        help="Fraction of bigrams that repeat — higher = more degenerate",
    )
    score_col2.text_input(
        "Kept-set size",
        value=str(display_step.get("kept_count", "")) if display_step else "",
        disabled=True,
        help="Number of tokens that survived the strategy's filter",
    )
