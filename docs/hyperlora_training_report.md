# HyperLoRA (Janus) — Architecture & Training Report

Complete technical reference for the HyperLoRA training and inference pipeline used by the Janus negotiation agent. Covers model architecture, dataset generation, prompt format, training loop, regularization, checkpoint format, and inference contract.

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [HyperLoRA Architecture (Code-Level)](#2-hyperlora-architecture-code-level)
3. [Dataset Generation Flow](#3-dataset-generation-flow)
4. [Canonical Prompt Format](#4-canonical-prompt-format)
5. [Training Pipeline](#5-training-pipeline)
6. [Regularization](#6-regularization)
7. [Checkpoint Format](#7-checkpoint-format)
8. [Inference & Runtime Usage](#8-inference--runtime-usage)
9. [End-to-End Flow Summary](#9-end-to-end-flow-summary)
10. [Files Referenced](#10-files-referenced)

---

## 1. Conceptual Overview

HyperLoRA is a **conditional LoRA** system. It keeps a base LLM (Qwen2-7B) entirely frozen and adds a low-rank update to selected linear layers — but *gates* that update with a hypernetwork conditioned on a scalar control variable $\rho$.

### 1.1 Core Equation

At every targeted linear layer, the effective weight is:

$$
W = W_{\text{base}} + \frac{\alpha}{r}\; B \;\mathrm{diag}\!\bigl(g(\rho)\bigr)\; A
$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| $W_{\text{base}}$ | $d_{\text{out}} \times d_{\text{in}}$ | Frozen pretrained weight matrix |
| $A$ | $r \times d_{\text{in}}$ | Trainable LoRA down-projection |
| $B$ | $d_{\text{out}} \times r$ | Trainable LoRA up-projection |
| $g(\rho)$ | $r$ | Gating vector from hypernetwork (per-rank element scaling) |
| $r$ | scalar | LoRA rank (default 16) |
| $\alpha$ | scalar | LoRA scaling factor (default 32.0) |

Because the gating vector $g(\rho)$ changes continuously with $\rho$, a **single set of trained weights** produces a full spectrum of behaviors at inference time.

### 1.2 Rho ($\rho$) Semantics

| Value | Meaning |
|-------|---------|
| $\rho \in [0, 1]$ | Normalized settlement point within the ZOPA: $\rho = \frac{\text{FinalPrice} - \text{SellerMin}}{\text{BuyerMax} - \text{SellerMin}}$ |
| $\rho \to 0$ | Buyer-favorable outcome (low price) |
| $\rho \to 1$ | Seller-favorable outcome (high price) |
| $\rho = -1.0$ | Impasse / failure — no agreement reached |

The model never sees an explicit buyer/seller role token. Instead, $\rho$ alone determines negotiation posture: the same weight matrices, gated differently by $g(\rho)$, produce buyer-like or seller-like behavior.

---

## 2. HyperLoRA Architecture (Code-Level)

All architecture code lives in [src/training/hyper_lora.py](../src/training/hyper_lora.py) (241 lines, never modified by training-script changes).

### 2.1 `RhoHyperNet` — Hypernetwork

Maps scalar $\rho$ → gating vector $g(\rho) \in \mathbb{R}^r$.

**Forward path:**

```
ρ [B, 1]
  │
  ├─ (optional) Fourier encode: sin(ρ·f), cos(ρ·f), [ρ]  →  [B, 2F+1]
  │      where f = 2^i · π for i = 0..F-1
  │
  ▼
Linear(input_dim, hidden)  →  SiLU
Linear(hidden, hidden)     →  SiLU
Linear(hidden, rank)       →  Sigmoid
  │
  ▼
g(ρ) [B, rank]    values in (0, 1)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 64 | Width of each MLP hidden layer |
| `use_fourier` | `False` | Fourier positional encoding of ρ |
| `fourier_freqs` | 8 | Number of frequency bands (if Fourier enabled) |
| `include_raw` | `True` | Concatenate raw ρ alongside Fourier features |
| `activation` | `sigmoid` | Final activation; alternatives: `tanh`, `softplus` |

**Key properties:**
- Sigmoid output bounds each gate element to $(0, 1)$, so each rank component is smoothly scaled rather than hard-selected.
- Fourier encoding gives the MLP higher-frequency access to the $\rho$ input, useful if the behavior landscape is non-monotonic.

### 2.2 `HyperLoRALinear` — Adapted Layer Wrapper

Replaces each targeted `nn.Linear` in the base model. Forward pass:

```
x [B, seq, d_in]
  │
  ├──► base_layer(x)  →  y_base [B, seq, d_out]     (frozen, no grad)
  │
  ├──► ρ = rho_getter()                               (closure → model.current_rho)
  │     │
  │     ▼
  │    hyper_net(ρ)  →  g [B, rank]
  │
  ├──► F.linear(x, lora_A)  →  z [B, seq, rank]      (down-project)
  │     │
  │     ▼
  │    z * g.unsqueeze(1)   →  z_gated [B, seq, rank] (element-wise gate)
  │     │
  │     ▼
  │    F.linear(z_gated, lora_B)  →  delta [B, seq, d_out]  (up-project)
  │     │
  │     ▼
  │    dropout(delta) * (α / r)  →  scaled_delta
  │
  ▼
y = y_base + scaled_delta
```

**Trainable parameters per layer:**
- `lora_A`: $r \times d_{\text{in}}$ — initialized with Kaiming uniform
- `lora_B`: $d_{\text{out}} \times r$ — initialized to zeros (so LoRA delta starts at zero)
- `hyper_net`: three-layer MLP (shared `RhoHyperNet` instance per layer)

**Frozen parameters:** all of `base_layer` (original `nn.Linear` weights + bias).

### 2.3 `inject_hyperlora` — Module Injection

Called once at model load time. Performs in-place surgery on the base model:

1. Attaches `model.current_rho = None` attribute.
2. Creates a `rho_getter` closure: `lambda: model.current_rho`.
3. Walks `model.named_modules()`, collecting every `nn.Linear` whose name ends with a target suffix.
4. Replaces each match with a new `HyperLoRALinear` wrapper (passing the original linear as `base_layer`).
5. Freezes **all** parameters, then un-freezes only those containing `"lora_"` or `"hyper_net"` in their name.

**Default target modules** (Qwen2 attention + MLP):
```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

For Qwen2-7B with rank 16, this typically yields ~60 M trainable parameters out of ~7.6 B total (~0.8%).

---

## 3. Dataset Generation Flow

### 3.1 Negotiation Episodes → JSONL

Dataset generation runs through [negotiation.py](../negotiation.py) in `--dataset_mode`. Each episode:
1. Samples `BuyerMax ~ N(900, 50)` and sets `SellerMin = BuyerMax - 500` (fixed ZOPA width).
2. Two deterministic agents (randomly selected from the strategy registry) negotiate for up to `max_turns` rounds.
3. The `DatasetWriter` ([src/logging/dataset_writer.py](../src/logging/dataset_writer.py)) records every step as a JSONL row.

**Rho calculation** (performed by `DatasetWriter`):
- If the episode ends in agreement at `FinalPrice`:
  $$\rho = \frac{\text{FinalPrice} - \text{SellerMin}}{\text{BuyerMax} - \text{SellerMin}}$$
- The same $\rho$ value is stamped onto **every step** in that trajectory.
- If no agreement is reached, the trajectory is saved with $\rho = -1.0$.

### 3.2 JSONL → Parquet

[src/data_prep/prepare_data.py](../src/data_prep/prepare_data.py) transforms raw JSONL into a training-ready `decision_steps.parquet`:

1. **Group** JSONL entries by `trajectory_id`.
2. **Label** each trajectory: success (any `ACCEPT` action) or failure (turns == max_turns with no accept).
3. **Assign** `rho_train`:
   - Success trajectories → the computed $\rho$ from the JSONL.
   - Failure trajectories → $\rho = -1.0$.
4. **Build per-step rows** with columns: `rho_train`, `turn`, `max_turns`, `turns_remaining`, `price_low`, `price_high`, `reservation_price`, `last_offer_price`, `history_roles` (list), `history_prices` (list), `target_action`, `target_price`.

### 3.3 Pre-Filtering at Training Time

The `NegotiationDataset` class in the training script applies one additional filter: rows where `price_high - price_low <= 1e-6` (degenerate ZOPA) are dropped and logged as warnings. This prevents division-by-zero in `normalize_price()`.

---

## 4. Canonical Prompt Format

A **single `build_prompt()` function** in [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py) defines the prompt layout. The inference agent ([src/agents/janus_agent.py](../src/agents/janus_agent.py)) **imports and calls the same function**, guaranteeing byte-identical formatting at training and inference.

### 4.1 Special Tokens

These tokens are added to the tokenizer via `add_special_tokens()` before training. They get dedicated token IDs so the model can attend to them as structural markers rather than arbitrary text.

| Token | Purpose |
|-------|---------|
| `<RHO>` | Precedes the numeric rho value (or `<RHO_FAIL>`) |
| `<RHO_FAIL>` | Stands in for the numeric rho value when $\rho = -1.0$ |
| `<SUCCESS>` | Binary success indicator: `0` or `1` |
| `<TURN>` | Current turn counter |
| `<TURNS_REMAINING>` | Turns left in the negotiation |
| `<RESERVATION_NORM>` | Agent's reservation price, normalized to $[0, 1]$ |
| `<LAST_OFFER_NORM>` | Most recent offer (normalized), or `NA` if no offers yet |
| `<HISTORY_LEN>` | Count of real (non-padding) entries in the history window |
| `<HISTORY>` | Fixed-width offer history (K slots) |
| `<INSTRUCTION>` | Begins the action instruction block |
| `<OUTPUT>` | Marks where the model's response begins (target tokens follow) |

### 4.2 Prompt Layout

**Success trajectory** ($\rho \in [0, 1]$):
```
<RHO> 0.5300
<SUCCESS> 1
<TURN> 7 / 20
<TURNS_REMAINING> 13
<RESERVATION_NORM> 0.8421
<LAST_OFFER_NORM> 0.4123
<HISTORY_LEN> 3
<HISTORY> EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY buyer:0.3000 seller:0.4000 buyer:0.4200
<INSTRUCTION> Output exactly one of:
ACCEPT
OFFER <PRICE_NORM>
<OUTPUT>
```

**Failure trajectory** ($\rho = -1.0$):
```
<RHO> <RHO_FAIL>
<SUCCESS> 0
<TURN> 20 / 20
<TURNS_REMAINING> 0
<RESERVATION_NORM> 0.7500
<LAST_OFFER_NORM> 0.3200
<HISTORY_LEN> 8
<HISTORY> buyer:0.1000 seller:0.9000 buyer:0.1500 seller:0.8500 buyer:0.2000 seller:0.8000 buyer:0.2500 seller:0.7500
<INSTRUCTION> Output exactly one of:
ACCEPT
OFFER <PRICE_NORM>
<OUTPUT>
```

**Target** is appended immediately after `<OUTPUT>\n`. It is either:
- `ACCEPT` (agreement action)
- `OFFER {price_norm:.4f}` (counter-offer, normalized to $[0, 1]$)

followed by the tokenizer's EOS token.

### 4.3 History Encoding

The `build_history_str()` function produces a fixed-width history window:

1. Take the **last K** entries from the full offer history (K defaults to 8, configurable via `--k_history`).
2. Each real entry is formatted as `{side}:{price_norm:.4f}` (e.g., `buyer:0.3000`, `seller:0.7500`).
3. If fewer than K real entries exist, **prepend** `EMPTY:EMPTY` padding slots until exactly K slots are filled.
4. Return the space-joined string and the count of real entries (`history_len`).

**Example with 3 real entries and K=8:**
```
EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY EMPTY:EMPTY buyer:0.3000 seller:0.4000 buyer:0.4200
```

The `EMPTY:EMPTY` sentinel was chosen over numeric padding (e.g., `PAD:0.0000`) to avoid the model interpreting padding as actual zero-price offers.

### 4.4 Helper Functions (Single Source of Truth)

All prompt-building helpers live in `train_janus_hyperlora.py` and are imported by `janus_agent.py`:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `format_rho_text(rho_value)` | `float → str` | Formats rho as `"<RHO> 0.5300"` or `"<RHO> <RHO_FAIL>"`. The **only** place textual rho is generated. |
| `format_success_text(rho_value)` | `float → str` | Returns `"<SUCCESS> 1"` or `"<SUCCESS> 0"` based on whether rho indicates failure. |
| `build_history_str(roles, prices, p_low, p_high, k)` | `→ (str, int)` | Normalizes prices, pads to K slots, returns `(history_text, num_real)`. |
| `build_prompt(rho, turn, max_turns, ...)` | `→ str` | Assembles the complete canonical prompt string. |
| `normalize_price(val, low, high)` | `float → float` | Maps raw price to $[0, 1]$, clamped. Returns 0.0 on degenerate ranges. |

### 4.5 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No `<ROLE>` token** | The model learns buyer/seller behavior entirely from $\rho$. Removing role prevents the model from shortcutting (ignoring rho and relying on role text). |
| **`<SUCCESS>` flag** | Gives the model semantic grounding for failure trajectories. Without it, the model sees $\rho = -1$ but has no textual signal that the negotiation ended in impasse. |
| **`EMPTY:EMPTY` padding** | Avoids the model learning that early history slots always contain `0.0000`, which could bias price predictions downward. A clearly non-numeric sentinel lets the model distinguish padding from data. |
| **`<HISTORY_LEN>`** | Explicit slot-occupancy count lets the model learn different strategies for early-game (sparse history) vs. late-game (full history window). |
| **`.4f` rho formatting** | 4 decimal places prevent floating-point representation drift between training and inference. A single `format_rho_text()` function is the sole source of truth. |

---

## 5. Training Pipeline

Training is implemented in [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py) (626 lines).

### 5.1 Model Setup

1. **Load tokenizer** from `--model_name` (default: `Qwen/Qwen2-7B`).
2. **Add special tokens** via `tokenizer.add_special_tokens()` — the 11 tokens in `SPECIAL_TOKENS`.
3. **Load base model** — optionally in 4-bit via `BitsAndBytesConfig` if `--use_qlora` is set.
4. **Resize token embeddings** to accommodate the new special tokens: `model.resize_token_embeddings(len(tokenizer))`.
5. **Inject HyperLoRA** via `inject_hyperlora(model, ...)` — replaces 7 target linear modules per transformer layer.
6. **Move to GPU** (skipped when QLoRA is active since `device_map` handles placement).

### 5.2 `NegotiationDataset.__getitem__`

For each parquet row:

1. Read scalars: `rho_train`, `turn`, `max_turns`, `turns_remaining`, `price_low`, `price_high`, `reservation_price`, `last_offer_price`.
2. Compute `reservation_norm = normalize_price(reservation_price, p_low, p_high)`.
3. Compute `last_offer_norm_str`: `"NA"` if null, else `f"{normalize_price(last_offer_price, p_low, p_high):.4f}"`.
4. Build history: `build_history_str(history_roles, history_prices, p_low, p_high, k_history)` → `(hist_str, hist_len)`.
5. Build target: `"ACCEPT"` or `f"OFFER {normalize_price(target_price, p_low, p_high):.4f}"`.
6. Build prompt: `build_prompt(rho, turn, max_turns, turns_remaining, res_norm, last_offer_norm_str, hist_len, hist_str)`.
7. Concatenate: `full_text = prompt + target_str + tokenizer.eos_token`.
8. Tokenize and **mask labels**: set `labels[:prompt_len] = -100` so the loss only applies to target tokens.
9. Return `{ input_ids, attention_mask, labels, rho: tensor([rho]) }`.

### 5.3 `HyperLoRACollator`

Pads variable-length sequences within a batch:
- `input_ids` padded with `tokenizer.pad_token_id`
- `attention_mask` padded with `0`
- `labels` padded with `-100` (ignored by loss)
- `rho` stacked into `[B, 1]` tensor

### 5.4 Training Loop

```
for each batch:
    1. model.current_rho = batch["rho"].to(device)       # [B, 1]
    2. outputs = model(input_ids, attention_mask, labels)  # causal LM forward
    3. lm_loss = outputs.loss / grad_accum
    4. total_loss = lm_loss

    5. if lambda_sep > 0:
         sep_loss = compute_gate_separation_loss(model, rho, exclude_failures)
         total_loss += (lambda_sep * sep_loss) / grad_accum

    6. total_loss.backward()

    7. if (step + 1) % grad_accum == 0:
         clip_grad_norm_(trainable_params, 1.0)
         optimizer.step()          # AdamW
         scheduler.step()          # linear warmup
         optimizer.zero_grad()

    8. if global_step % save_every == 0:
         save_hyperlora_adapter(...)
```

**Optimizer:** AdamW with `--lr` (default 2e-4).
**Schedule:** Linear warmup over `--warmup_steps` (default 500), then linear decay to 0 over `--max_steps` (default 20,000).
**Gradient accumulation:** effective batch size = `batch_size × grad_accum` (default 4 × 8 = 32).
**Gradient clipping:** max norm 1.0.

### 5.5 Data Split

- 99% train, 1% eval (shuffled, seeded).
- Eval split is reserved but not used in the current loop (placeholder for future eval-loss logging).

---

## 6. Regularization

### 6.1 Gate Separation Loss (`--lambda_sep`)

**Purpose:** Prevent mode collapse where different $\rho$ values produce identical gating vectors (i.e., the model ignores $\rho$).

**Mechanism** (in `compute_gate_separation_loss`):
1. Given batch rho tensor $[\rho_1, \rho_2, \ldots, \rho_B]$, create a random permutation $[\rho_{\pi(1)}, \ldots, \rho_{\pi(B)}]$.
2. For each `HyperLoRALinear` module, compute:
   - $g_i = \text{hyper\_net}(\rho_i)$ and $g_{\pi(i)} = \text{hyper\_net}(\rho_{\pi(i)})$
   - $D = \frac{1}{L}\sum_{\ell=1}^{L} \mathbb{E}_i\left[\|g_i^{(\ell)} - g_{\pi(i)}^{(\ell)}\|^2\right]$ (average L2 distance across all layers)
3. Compute a distance-based weight: $w = \mathbb{E}_i\left[\min(|\rho_i - \rho_{\pi(i)}|, 1)\right]$.
4. Return $-D \cdot w$ (negative because the optimizer minimizes: maximizing distance between different-$\rho$ gates).

**Failure exclusion** (`--sep_exclude_failures true`): Rows with $\rho = -1.0$ are removed from the batch before computing this loss. Rationale: $\rho = -1$ is semantically distinct from the $[0,1]$ continuum, and we don't want the model penalized for mapping $-1$ far from $0.5$ (which is the desired behavior).

### 6.2 Smoothness Loss (`--lambda_smooth`)

**Purpose:** Ensure nearby $\rho$ values produce smoothly varying gating vectors (prevent erratic jumps in behavior for small $\rho$ changes).

Currently exposed as a CLI argument with default 0.0 (off).
**Note:** This feature is currently **reserved** and not applied in the training loop.

---

## 7. Checkpoint Format

`save_hyperlora_adapter` writes three artifacts to the checkpoint directory:

### 7.1 `adapter_state.pt`
A `state_dict` containing **only trainable parameters** (LoRA A/B matrices and hypernetwork weights). All frozen base-model weights are excluded. Typical size for rank-16 Qwen2-7B: ~240 MB.

### 7.2 `adapter_config.json`
```json
{
  "rank": 16,
  "alpha": 32.0,
  "dropout": 0.05,
  "hyper_hidden": 64,
  "base_model": "Qwen/Qwen2-7B",
  "k_history": 8,
  "augmented_tokens": ["<RHO>", "<RHO_FAIL>", "<SUCCESS>", "<TURN>", ...]
}
```

The `k_history` field is read by the inference agent to set the correct history window size without hardcoding.

### 7.3 Tokenizer Files
Saved via `tokenizer.save_pretrained(output_dir)`. Includes the added special tokens so the inference agent can load the tokenizer from the adapter path directly.

---

## 8. Inference & Runtime Usage

Implemented in [src/agents/janus_agent.py](../src/agents/janus_agent.py) (276 lines).

### 8.1 Model Loading (`JanusAgent.__init__`)

1. **Load tokenizer** from `adapter_path` (falls back to `model_path` on failure).
2. **Read `adapter_config.json`** from `adapter_path`.
3. **Load base model** from `model_path` (e.g., `Qwen/Qwen2-7B-Instruct`).
4. **Inject HyperLoRA** using config values (`rank`, `alpha`, `hyper_hidden`, `target_modules`).
5. **Load `adapter_state.pt`** via `model.load_state_dict(state_dict, strict=False)`.
6. Set `model.eval()`.
7. Cache the `(model, tokenizer)` pair globally to avoid reloading for multiple agents.

### 8.2 Prompt Construction (`generate_response`)

The inference agent imports and calls the **same canonical helpers** from the training module:

```python
from src.training.train_janus_hyperlora import (
    normalize_price, build_history_str, build_prompt, SPECIAL_TOKENS,
)
```

Steps:
1. Extract context: `turn`, `max_turns`, `reservation`, `history` from `domain_private_context`.
2. `res_norm = normalize_price(reservation, p_low, p_high)`.
3. `history_str, history_len = build_history_str(h_roles, h_prices, p_low, p_high, k)` where `k = adapter_config.get("k_history", 8)`.
4. `last_offer_norm_str` = `"NA"` or `f"{normalize_price(last_price, p_low, p_high):.4f}"`.
5. `prompt = build_prompt(rho=self.rho, turn=turn, max_turns=max_turns, ...)`.
6. Set `model.current_rho = tensor([[self.rho]])` — the **same numeric value** that generated the textual rho in the prompt.
7. `model.generate(**inputs, max_new_tokens=20, do_sample=False)`.
8. Parse output: extract `ACCEPT` or `OFFER {price_norm}`, denormalize price to real domain.

### 8.3 Key Contract — Three Synchronized Signals

The model expects three mutually consistent inputs:

| Signal | Source | Description |
|--------|--------|-------------|
| **Textual ρ** | `format_rho_text(self.rho)` inside `build_prompt()` | String in the prompt, e.g., `<RHO> 0.5300` |
| **Numeric ρ tensor** | `model.current_rho = tensor([[self.rho]])` | Feeds every `RhoHyperNet` to produce $g(\rho)$ |
| **`<SUCCESS>` flag** | `format_success_text(self.rho)` inside `build_prompt()` | `1` if $\rho \in [0,1]$, `0` if $\rho = -1$ |

All three derive from the **same `self.rho` value**, making divergence impossible.

### 8.4 Output Parsing

The generated text (after the prompt) is parsed as:
- If `"ACCEPT"` appears → return `PriceAction(type="ACCEPT", price=None)`
- If `OFFER {float}` matches → denormalize: `price = p_low + val_norm * (p_high - p_low)`, return `PriceAction(type="OFFER", price=price)`
- Fallback → offer at reservation price

---

## 9. End-to-End Flow Summary

```
┌─────────────────────────────────────────────────────────┐
│  1. Generate Episodes                                    │
│     negotiation.py --dataset_mode                        │
│     Deterministic agents play N episodes                 │
│                          │                               │
│                          ▼                               │
│  2. Write JSONL                                          │
│     DatasetWriter computes ρ per trajectory              │
│     → datasets/price_domain.jsonl                        │
│                          │                               │
│                          ▼                               │
│  3. Prepare Parquet                                      │
│     prepare_data.py: group by trajectory, label ρ,       │
│     build per-step training rows                         │
│     → datasets/processed_tables/decision_steps.parquet   │
│                          │                               │
│                          ▼                               │
│  4. Train HyperLoRA                                      │
│     train_janus_hyperlora.py:                            │
│       • Load Qwen2-7B + inject HyperLoRA                │
│       • Add 11 special tokens                            │
│       • For each step: build_prompt → tokenize →         │
│         mask labels → forward → LM loss +                │
│         optional separation/smoothness reg →             │
│         backward → AdamW step                            │
│                          │                               │
│                          ▼                               │
│  5. Save Checkpoint                                      │
│     adapter_state.pt + adapter_config.json + tokenizer   │
│     → checkpoints/janus_v1/final/                        │
│                          │                               │
│                          ▼                               │
│  6. Inference                                            │
│     JanusAgent loads adapter, imports build_prompt,      │
│     sets model.current_rho, generates, parses output     │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Files Referenced

| File | Role |
|------|------|
| [src/training/hyper_lora.py](../src/training/hyper_lora.py) | Core architecture: `RhoHyperNet`, `HyperLoRALinear`, `inject_hyperlora` |
| [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py) | Training script + canonical prompt builder (`build_prompt`, `format_rho_text`, `build_history_str`, `normalize_price`, `compute_gate_separation_loss`) |
| [src/agents/janus_agent.py](../src/agents/janus_agent.py) | Inference agent — imports canonical helpers from training module |
| [src/data_prep/prepare_data.py](../src/data_prep/prepare_data.py) | JSONL → Parquet transformation |
| [src/logging/dataset_writer.py](../src/logging/dataset_writer.py) | JSONL dataset writer with ρ computation |
| [negotiation.py](../negotiation.py) | Main negotiation engine / episode runner |
| [config/settings.py](../config/settings.py) | Global defaults (`K_HISTORY=8`, price ranges, model paths) |
