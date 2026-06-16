# 🧠 Neural Text Decoding Playground

> An interactive, fully transparent playground for exploring neural text **decoding strategies** — built to reproduce the core ideas of Holtzman et al., 2020, *["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751)*.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/Model-GPT--2%20%2F%20PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CPU](https://img.shields.io/badge/Runs%20on-CPU-success)

Generate text **one token at a time** with GPT-2 (124M) and see *exactly* why each token was chosen — which candidates existed, which were kept or cut by the decoding strategy, and which one was finally sampled.

---

## ✨ Features

- **Six decoding strategies** — Greedy, Beam Search, Pure Sampling, Temperature Sampling, Top-k, and Nucleus (Top-p) — all switchable live.
- **Step-by-step generation** — click **Step Once** to watch a single token get chosen, with a color-coded bar chart of the top-15 candidates.
  - 🟩 GREEN = kept by the strategy's filter
  - 🟥 RED = cut by the filter
  - 🟨 GOLD = the token that was actually sampled
- **Batch generation + replay** — click **Generate N** to produce many tokens at once, then drag the **Replay step** slider to scrub back through any step of that run.
- **PEAKED / FLAT distribution badge** — a plain-English read on how confident the model is at each step, and why fixed-k strategies struggle where nucleus sampling adapts.
- **Repetition score** — a live bigram-repetition metric that quantifies the degenerate, looping text that greedy/beam decoding tends to produce.
- **Reproducible runs** — a fixed random seed makes every sampling run repeatable.

## 📐 How each strategy works

| Strategy | Idea | Trade-off |
|---|---|---|
| **Greedy** | Always picks the single highest-probability token. | Fast & deterministic, but repeats itself a lot. |
| **Beam Search** | Tracks the best *b* sequences in parallel. | Better than greedy, but still degenerates on long, open-ended text. |
| **Pure Sampling** | Samples from the entire vocabulary as-is. | No repetition, but occasionally picks incoherent tokens from the tail. |
| **Temperature** | Scales logits by `1/T` before sampling. | `T<1` sharpens (safer), `T>1` flattens (more random) — doesn't fix the tail-mass problem. |
| **Top-k** | Samples only from the *k* highest-probability tokens. | Simple, but a fixed *k* doesn't adapt to how confident the model is. |
| **Nucleus (Top-p)** | Keeps the *smallest* set of tokens whose cumulative probability ≥ *p*. | Adapts: shrinks when the model is confident, grows when it's uncertain — the paper's main contribution. |

## 🗂️ Project structure

```
decoding_playground/
├── app.py              # Streamlit UI — layout, charts, callbacks (no model logic)
├── decoder.py           # Pure backend — model loading, decoding strategies, metrics
├── requirements.txt      # Python dependencies
└── README.md
```

The split is deliberate: `decoder.py` has zero UI code and can be run standalone (`python decoder.py`) to sanity-check the decoding logic without touching the browser.

## 🚀 Setup & installation

### 1. Clone the repository

```bash
git clone https://github.com/nishchalbasyal/Neural_Text_Decoding_Visualizer.git
cd Neural_Text_Decoding_Visualizer/decoding_playground
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

(Python 3.10+ recommended. Everything runs on CPU — no GPU required. GPT-2 weights download automatically on first run.)

### 4. Launch the app

```bash
streamlit run app.py
```

A browser tab opens automatically at **http://localhost:8501**. The first click of **Step Once** or **Generate N** will be slower while GPT-2 loads — after that it's cached for the rest of the session.

### 5. (Optional) Verify the backend on its own

```bash
python decoder.py
```

Runs a quick smoke test of the decoding logic with no browser involved.

## 🕹️ Usage

1. Type a prompt (or keep the default unicorn story).
2. Pick a **Decoding Strategy** — the relevant slider (temperature / k / p / beam width) appears automatically.
3. Click **Step Once** to generate a single token and inspect the chart, badge, and explanation.
4. Click **Generate N** to generate several tokens at once, then drag **Replay step** to revisit any step.
5. Click **Reset** to clear the run and start over.

## 🧰 Tech stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[PyTorch](https://pytorch.org/) + [🤗 Transformers](https://huggingface.co/docs/transformers)** — GPT-2 model and tokenizer
- **[Plotly](https://plotly.com/python/)** — interactive probability charts

## 📖 Reference

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
