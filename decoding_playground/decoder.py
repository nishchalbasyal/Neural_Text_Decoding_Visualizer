# decoder.py - GPT-2 model loading, the six decoding strategies, and scoring. No UI code.

import math
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_tokenizer: GPT2Tokenizer | None = None
_model: GPT2LMHeadModel | None = None


# loads GPT-2 once and caches it on the module
'''
This function loads GPT-2 and its tokenizer. The tokenizer converts text into tokens that GPT-2 understands. The model is loaded only once and then cached, which improves efficiency.
'''

def load_model() -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
    return _tokenizer, _model


'''
This function takes input text and performs one forward pass through GPT-2. It returns logits, which are raw scores for every possible next token.
'''

# runs one forward pass, returns raw logits for the next token
def get_next_token_logits(text: str) -> torch.Tensor:
    tokenizer, model = load_model()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits[0, -1, :] #This line extracts the prediction for the next token.


# applies one decoding strategy to raw logits; "beam" here is a single-step preview, real beam search runs in _generate_beam

# This is the core function of the project. It implements 4 decoding strategies
def apply_strategy(
    logits: torch.Tensor,
    strategy: str,
    temperature: float = 1.0,
    k: int = 50,
    p: float = 0.9,
    beam_width: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    vocab_size = logits.shape[0]


   # Greedy always selects the highest-probability token. It is simple but often generates repetitive text.
    if strategy == "greedy":
        probs = F.softmax(logits, dim=-1)
        best_idx = int(probs.argmax())
        kept_mask = torch.zeros(vocab_size, dtype=torch.bool)
        kept_mask[best_idx] = True
        probs_after = probs * kept_mask.float()
        info = {
            "kept_count": 1,
            "cumulative_mass": float(probs[best_idx]),
        }

    elif strategy == "beam":
        probs = F.softmax(logits, dim=-1)
        actual_b = min(beam_width, vocab_size)
        top_indices = probs.topk(actual_b).indices
        kept_mask = torch.zeros(vocab_size, dtype=torch.bool)
        kept_mask[top_indices] = True
        probs_after = probs * kept_mask.float()
        mass = float(probs_after.sum())
        probs_after = probs_after / (mass + 1e-12)
        info = {"kept_count": actual_b, "cumulative_mass": mass}

    # elif strategy == "pure":
    #     probs = F.softmax(logits, dim=-1)
    #     kept_mask = torch.ones(vocab_size, dtype=torch.bool)
    #     probs_after = probs
    #     info = {"kept_count": vocab_size, "cumulative_mass": 1.0}

    # elif strategy == "temperature":
    #     scaled = logits / max(temperature, 1e-8)
    #     probs = F.softmax(scaled, dim=-1)
    #     kept_mask = torch.ones(vocab_size, dtype=torch.bool)
    #     probs_after = probs
    #     info = {
    #         "kept_count": vocab_size,
    #         "cumulative_mass": 1.0,
    #         "temperature": temperature,
    #     }

    elif strategy == "topk":
        probs = F.softmax(logits, dim=-1)
        actual_k = min(k, vocab_size)
        top_values, top_indices = probs.topk(actual_k)
        kept_mask = torch.zeros(vocab_size, dtype=torch.bool)
        kept_mask[top_indices] = True
        probs_after = probs * kept_mask.float()
        mass = float(probs_after.sum())
        probs_after = probs_after / (mass + 1e-12)
        info = {"kept_count": actual_k, "cumulative_mass": mass}

    elif strategy == "nucleus":
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cum_probs = sorted_probs.cumsum(dim=0)
        past_cutoff = (cum_probs - sorted_probs) >= p
        sorted_probs[past_cutoff] = 0.0
        kept_mask = torch.zeros(vocab_size, dtype=torch.bool)
        kept_mask[sorted_indices] = sorted_probs > 0.0
        probs_after = probs * kept_mask.float()
        mass = float(probs_after.sum())
        probs_after = probs_after / (mass + 1e-12)
        info = {
            "kept_count": int(kept_mask.sum()),
            "cumulative_mass": mass,
            "p": p,
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return probs_after, kept_mask, info


# picks a token id from the filtered distribution; greedy uses argmax, everything else samples
def pick_token(
    probs_after: torch.Tensor,
    strategy: str,
    seed: int = 42,
) -> int:
    torch.manual_seed(seed)
    if strategy == "greedy":
        return int(probs_after.argmax())
    return int(torch.multinomial(probs_after, num_samples=1))


# labels a distribution PEAKED or FLAT from top-token probability / entropy
def classify_distribution(probs: torch.Tensor) -> tuple[str, float]:
    top_prob = float(probs.max())
    if top_prob > 0.6:
        return "PEAKED", top_prob

    entropy = float(-(probs * (probs + 1e-12).log()).sum())
    h_max = math.log(probs.shape[0])
    norm_entropy = entropy / h_max

    label = "FLAT" if norm_entropy > 0.5 else "PEAKED"
    return label, top_prob


# generates one next token and builds the step_data dict the UI renders
def step_once(
    text: str,
    strategy: str,
    params: dict,
    seed: int = 42,
) -> tuple[str, dict]:
    tokenizer, _ = load_model()
    if not text.strip():
        text = "The"

    logits = get_next_token_logits(text)

    probs_after, kept_mask, info = apply_strategy(
        logits,
        strategy,
        temperature=params.get("temperature", 1.0),
        k=params.get("k", 50),
        p=params.get("p", 0.9),
        beam_width=params.get("beam_width", 4),
    )

    token_id = pick_token(probs_after, strategy, seed)
    token_str = tokenizer.decode([token_id])

    raw_probs = F.softmax(logits, dim=-1)
    top15_indices = raw_probs.topk(15).indices.tolist()

    candidates = [
        {
            "token": tokenizer.decode([idx]),
            "prob": float(raw_probs[idx]),
            "kept": bool(kept_mask[idx]),
            "chosen": (idx == token_id),
        }
        for idx in top15_indices
    ]

    dist_label, dist_metric = classify_distribution(raw_probs)

    explanation = _build_explanation(
        strategy, token_str, info, probs_after, token_id, params
    )

    step_data = {
        "candidates": candidates,
        "chosen_token": token_str,
        "chosen_token_id": token_id,
        "kept_count": int(info["kept_count"]),
        "cumulative_mass": float(info.get("cumulative_mass", 1.0)),
        "dist_label": dist_label,
        "dist_metric": dist_metric,
        "explanation": explanation,
        "strategy": strategy,
    }
    return token_str, step_data


# one-line explanation of why this token was picked
def _build_explanation(
    strategy: str,
    token_str: str,
    info: dict,
    probs_after: torch.Tensor,
    token_id: int,
    params: dict,
) -> str:
    chosen_prob = float(probs_after[token_id])
    t = token_str.strip() or repr(token_str)
    kept_n = int(info["kept_count"])
    mass_pct = info.get("cumulative_mass", 1.0) * 100

    if strategy == "greedy":
        return (
            f"Greedy: always picks the highest-probability token → "
            f"'{t}' (p={chosen_prob:.3f})"
        )
    if strategy == "beam":
        return (
            f"Beam kept top-{kept_n} tokens ({mass_pct:.1f}% mass); "
            f"sampled '{t}' (p={chosen_prob:.3f})"
        )
    if strategy == "pure":
        return (
            f"Pure sampling: all {kept_n} vocab tokens eligible; "
            f"landed on '{t}' (p={chosen_prob:.4f})"
        )
    if strategy == "temperature":
        T = params.get("temperature", 1.0)
        direction = "sharpened" if T < 1 else ("flattened" if T > 1 else "unchanged")
        return (
            f"Temperature {T:.2f} {direction} the distribution; "
            f"sampled '{t}' (p={chosen_prob:.3f})"
        )
    if strategy == "topk":
        return (
            f"Top-{kept_n} kept {kept_n} tokens ({mass_pct:.1f}% mass); "
            f"sampled '{t}' (p={chosen_prob:.3f})"
        )
    if strategy == "nucleus":
        p_val = params.get("p", 0.9)
        return (
            f"Nucleus (p={p_val}) kept {kept_n} tokens "
            f"({mass_pct:.1f}% mass); sampled '{t}' (p={chosen_prob:.3f})"
        )
    return f"Sampled '{t}' (p={chosen_prob:.3f})"


# generates n_tokens tokens by calling step_once in a loop (beam search is handled separately)
def generate(
    prompt: str,
    strategy: str,
    params: dict,
    n_tokens: int = 50,
    seed: int = 42,
) -> tuple[str, list[dict]]:
    if strategy == "beam":
        return _generate_beam(prompt, params, n_tokens, seed)

    text = prompt
    steps: list[dict] = []
    for i in range(n_tokens):
        token_str, step_data = step_once(text, strategy, params, seed=seed + i)
        text += token_str
        steps.append(step_data)
    return text, steps


# beam search keeping beam_width full sequences, scored by summed log-prob
def _generate_beam(
    prompt: str,
    params: dict,
    n_tokens: int,
    seed: int,
) -> tuple[str, list[dict]]:
    tokenizer, _ = load_model()
    beam_width = params.get("beam_width", 4)

    beams: list[tuple[float, str, list[dict]]] = [(0.0, prompt, [])]

    for i in range(n_tokens):
        candidates: list[tuple[float, str, list[dict]]] = []

        for score, text, steps in beams:
            logits = get_next_token_logits(text)
            raw_probs = F.softmax(logits, dim=-1)

            top_probs, top_ids = raw_probs.topk(beam_width)
            for j in range(beam_width):
                tid = int(top_ids[j])
                tp = float(top_probs[j])
                new_score = score + math.log(tp + 1e-30)
                new_text = text + tokenizer.decode([tid])

                kept_mask = torch.zeros(raw_probs.shape[0], dtype=torch.bool)
                kept_mask[top_ids] = True
                top15_idx = raw_probs.topk(15).indices.tolist()
                cands = [
                    {
                        "token": tokenizer.decode([idx]),
                        "prob": float(raw_probs[idx]),
                        "kept": bool(kept_mask[idx]),
                        "chosen": (idx == tid),
                    }
                    for idx in top15_idx
                ]
                dist_label, dist_metric = classify_distribution(raw_probs)
                t = tokenizer.decode([tid]).strip() or repr(tokenizer.decode([tid]))
                explanation = (
                    f"Beam kept top-{beam_width} tokens; "
                    f"selected '{t}' (log-score={new_score:.2f})"
                )
                sd = {
                    "candidates": cands,
                    "chosen_token": tokenizer.decode([tid]),
                    "chosen_token_id": tid,
                    "kept_count": beam_width,
                    "cumulative_mass": float(top_probs.sum()),
                    "dist_label": dist_label,
                    "dist_metric": dist_metric,
                    "explanation": explanation,
                    "strategy": "beam",
                }
                candidates.append((new_score, new_text, steps + [sd]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    best_score, best_text, best_steps = beams[0]
    return best_text, best_steps


# fraction of repeated n-grams; higher means more degenerate text
def repetition_score(text: str, n: int = 2) -> float:
    tokenizer, _ = load_model()
    ids = tokenizer.encode(text)
    if len(ids) < n:
        return 0.0
    ngrams = [tuple(ids[i: i + n]) for i in range(len(ids) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts) if counts else 0.0


# manual smoke test: run `python decoder.py`
if __name__ == "__main__":
    PROMPT = "The unicorn lived in a magical forest and it had a beautiful silver horn"
    PARAMS = {"temperature": 1.0, "k": 50, "p": 0.9, "beam_width": 4}

    print("Loading GPT-2 (this may take a moment on first run)…")
    load_model()

    print("\n── Nucleus sampling, one step ─────────────────────────────")
    token_str, sd = step_once(PROMPT, "nucleus", PARAMS, seed=42)
    print(f"Chosen token : {token_str!r}")
    print(f"Explanation  : {sd['explanation']}")
    print(f"Distribution : {sd['dist_label']}  (top-token p={sd['dist_metric']:.3f})")
    print(f"Kept-set size: {sd['kept_count']} tokens")
    print("\nTop-5 candidates (from raw distribution):")
    for c in sd["candidates"][:5]:
        status = "CHOSEN" if c["chosen"] else ("kept " if c["kept"] else "CUT  ")
        print(f"  [{status}]  {c['token']!r:20s}  p={c['prob']:.4f}")

    print("\n── Greedy, one step ────────────────────────────────────────")
    token_str_g, sd_g = step_once(PROMPT, "greedy", PARAMS, seed=42)
    print(f"Chosen token : {token_str_g!r}")
    print(f"Explanation  : {sd_g['explanation']}")

    rep = repetition_score(PROMPT + token_str + token_str_g)
    print(f"\nRepetition score on combined text: {rep*100:.1f}%")
    print("\nSmoke test passed.")
