# Single Price Point (SPP) Negotiation Domain

A self-contained research framework for studying negotiation dynamics in single-issue price bargaining scenarios. This system supports both LLM-powered and deterministic agents, enabling comprehensive analysis of AI negotiation strategies.

## 🎯 Overview

This domain focuses on **single scalar price negotiations** where:
- A **Buyer** wants to minimize the price (maximize `BuyerMax - Price`)
- A **Seller** wants to maximize the price (maximize `Price - SellerMin`)
- The **ZOPA** (Zone of Possible Agreement) is defined as `[SellerMin, BuyerMax]`

### Agent Types

1. **Deterministic Agents**: Follow mathematical concession strategies (Boulware, Linear, Tit-for-Tat, etc.)
2. **LLM-Wrapped Deterministic Agents**: Deterministic backend with LLM natural language generation
3. **Free-form LLM Agents**: Pure LLM negotiators with system instructions but no forced output
4. **Janus (HyperLoRA) Agents**: Trained agents with continuously controllable behavior via a scalar ρ (rho) parameter

---

## 📂 Project Structure

```
Single Price Point Negotiation Domain/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── negotiation.py                   # Main entry point for negotiation sessions
├── run_janus_vs_deterministic.py    # Janus vs deterministic strategy benchmarks
├── visualize_concessions.py         # Strategy visualization tool
│
├── config/
│   ├── settings.py                  # Global configuration & defaults
│   └── *.txt                        # System instruction templates
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py            # Abstract base agent
│   │   ├── agent_factory.py         # Agent registry & instantiation
│   │   ├── price_strategies.py      # Deterministic concession strategies
│   │   ├── price_strategy_agent.py  # LLM wrapper for deterministic strategies
│   │   ├── basic_price_agent.py     # Free-form LLM agent
│   │   ├── janus_agent.py           # HyperLoRA inference agent (imports
│   │   │                            #   canonical helpers from training module)
│   │   └── ollama_agent.py          # Ollama LLM integration
│   │
│   ├── core/
│   │   └── price_structures.py      # PriceState, PriceAction dataclasses
│   │
│   ├── domain/
│   │   └── single_issue_price_domain.py
│   │
│   ├── training/
│   │   ├── hyper_lora.py            # HyperLoRA core: RhoHyperNet,
│   │   │                            #   HyperLoRALinear, inject_hyperlora
│   │   └── train_janus_hyperlora.py # Training script + canonical prompt
│   │                                #   builder (build_prompt, format_rho_text,
│   │                                #   build_history_str, etc.)
│   │
│   ├── data_prep/
│   │   └── prepare_data.py          # JSONL → Parquet conversion
│   │
│   ├── logging/
│   │   ├── csv_logger.py            # Experiment CSV logger
│   │   └── dataset_writer.py        # JSONL dataset writer (rho computation)
│   │
│   └── utils/
│       └── thinking_model_processor.py
│
├── datasets/                        # Generated datasets
│   ├── price_domain.jsonl
│   └── processed_tables/
│       └── decision_steps.parquet
│
├── checkpoints/                     # Trained model checkpoints
│   └── janus_v1/
│       └── final/
│           ├── adapter_state.pt
│           ├── adapter_config.json
│           └── tokenizer files
│
├── logs/                            # Experiment logs (CSV)
├── concession_plots/                # Visualization outputs
│
├── docs/
│   ├── hyperlora_training_report.md # Full HyperLoRA architecture & training report
│   └── price_strategies.md          # Deterministic strategy documentation
│
└── analysis/
    └── analyze_results_spp.ipynb
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd "Single Price Point Negotiation Domain"
pip install -r requirements.txt
```

### 2. Run a Basic Negotiation

**LLM vs Deterministic:**
```bash
python negotiation.py --buyer_type basic --seller_strategy boulware_linear --max_turns 5 --model_name qwen2:7b
```

**Deterministic vs Deterministic (No LLM):**
```bash
python negotiation.py --buyer_strategy boulware_conceding --seller_strategy hardliner --max_turns 10 --buyer_type deterministic --seller_type deterministic
```

### 3. Generate Training Dataset

```bash
python negotiation.py --num_runs 10000 --max_turns 20 \
  --dataset_out datasets/price_domain.jsonl
```

(Note: Dataset logging is enabled automatically when `--dataset_out` is provided)

### 4. Prepare Data → Train Janus (HyperLoRA)

```bash
# Step 1: Convert JSONL to Parquet
python src/data_prep/prepare_data.py \
  --jsonl_path datasets/price_domain.jsonl \
  --output_dir datasets/processed_tables

# Step 2: Train (single-GPU, optional QLoRA 4-bit quantization)
python src/training/train_janus_hyperlora.py \
  --decision_steps_path datasets/processed_tables/decision_steps.parquet \
  --output_dir checkpoints/janus_v1 \
  --model_name "Qwen/Qwen2-7B" \
  --k_history 8 \
  --use_qlora

# Optional: enable gate separation regularization
python src/training/train_janus_hyperlora.py \
  --decision_steps_path datasets/processed_tables/decision_steps.parquet \
  --output_dir checkpoints/janus_v1 \
  --model_name "Qwen/Qwen2-7B" \
  --lambda_sep 0.01 \
  --sep_exclude_failures true \
  --use_qlora
```

### 5. Run Janus vs Deterministic Benchmark

```bash
python run_janus_vs_deterministic.py
```

### 6. Analyze Results

Open `analysis/analyze_results_spp.ipynb` in Jupyter and run all cells.

---

## 🤖 Agent Types Reference

### Deterministic Strategies (STRATEGY_REGISTRY)

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| `boulware_very_conceding` | Rapidly concedes (β=0.2) | `beta`, `static_margin` |
| `boulware_conceding` | Moderate early concession (β=0.5) | `beta`, `static_margin` |
| `boulware_linear` | Linear concession (β=1.0) | `beta`, `static_margin` |
| `boulware_firm` | Slow concession (β=2.0) | `beta`, `static_margin` |
| `boulware_hard` | Very slow concession (β=4.0) | `beta`, `static_margin` |
| `price_fixed_strict` | Fixed margin (±20) | `margin` |
| `price_fixed_loose` | Fixed margin (±100) | `margin` |
| `tit_for_tat` | Mirror opponent concessions | `initial_margin` |
| `linear_standard` | Standard linear | `static_margin` |
| `split_difference` | Midpoint offers | `initial_margin` |
| `time_dependent` | Relaxing threshold | `margin` |
| `hardliner` | Hold until final round | `margin` |
| `random_zopa` | Random in ZOPA (oracle) | `zopa_min`, `zopa_max` |
| `micro_fine` | MiCRO (step=5) | `step_size` |
| `micro_moderate` | MiCRO (step=25) | `step_size` |
| `micro_coarse` | MiCRO (step=100) | `step_size` |

### LLM Agent Types

| Agent Type | Description |
|------------|-------------|
| `basic` / `basic_price` | Free-form LLM negotiator |
| `price_strategy` | LLM wrapper around deterministic strategy |
| `janus` | HyperLoRA-controlled agent (ρ ∈ [0, 1]) |

---

## 📊 Janus (HyperLoRA) Architecture

Janus uses **HyperLoRA** to enable continuous behavioral control over a single frozen base LLM (Qwen2-7B). At each linear layer targeted for adaptation:

$$W = W_{\text{base}} + \frac{\alpha}{r}\; B\;\mathrm{diag}(g(\rho))\; A$$

Where:
- $W_{\text{base}}$ — frozen pretrained weights
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ — trainable LoRA matrices
- $g(\rho) \in \mathbb{R}^{r}$ — gating vector produced by a hypernetwork MLP conditioned on scalar $\rho$
- $\rho \in [0, 1]$ — normalized settlement point within the ZOPA (0 = buyer-favorable, 1 = seller-favorable)
- $\rho = -1.0$ — failure/impasse mode (no agreement reached)

### Canonical Prompt Format

The model receives **identical prompt structure** at training and inference. A single `build_prompt()` function (defined in `train_janus_hyperlora.py`, imported at inference by `janus_agent.py`) is the sole source of truth:

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

**Key design choices:**
- **No `<ROLE>` token** — the model learns buyer vs. seller behavior entirely from the numeric ρ signal
- **`<SUCCESS>` flag** — `1` for agreements, `0` for failures; gives the model semantic grounding for impasse trajectories
- **`EMPTY:EMPTY` padding** — history is always exactly K slots (default 8); empty positions use `EMPTY:EMPTY` instead of numeric padding to avoid confusing the model with fake price data
- **`<HISTORY_LEN>`** — explicit count of real (non-padding) history entries
- **Textual ρ** formatted to `.4f` by `format_rho_text()`, ensuring training-inference parity
- **Failure trajectories** use `<RHO> <RHO_FAIL>` and `<SUCCESS> 0`

### Three Synchronized Signals

The model expects three signals to be consistent:
1. **Textual ρ** in the prompt (via `format_rho_text()`)
2. **Numeric ρ tensor** set on `model.current_rho` (feeds the hypernetwork)
3. **`<SUCCESS>` flag** consistent with the ρ value (1 if ρ ∈ [0,1], 0 if ρ = -1)

### Training CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2-7B` | HuggingFace model ID |
| `--decision_steps_path` | *(required)* | Path to `decision_steps.parquet` |
| `--output_dir` | *(required)* | Checkpoint output directory |
| `--k_history` | `8` | Number of fixed history slots (K) |
| `--include_failures` | `true` | Include ρ = −1.0 trajectories |
| `--use_qlora` | `false` | Enable 4-bit QLoRA quantization |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `8` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--max_steps` | `20000` | Total training steps |
| `--rank` | `16` | LoRA rank |
| `--alpha` | `32.0` | LoRA scaling factor |
| `--hyper_hidden` | `64` | Hypernetwork hidden dimension |
| `--lambda_sep` | `0.0` | Gate separation regularization weight (0 = off) |
| `--sep_exclude_failures` | `true` | Exclude ρ = −1 from separation loss |
| `--lambda_smooth` | `0.0` | Gate smoothness regularization weight (0 = off) |

See [docs/hyperlora_training_report.md](docs/hyperlora_training_report.md) for the full architecture and training flow report.

---

## 📈 CSV Log Format

All negotiations log to `logs/` with columns:

| Column | Description |
|--------|-------------|
| `session_id` | Unique session identifier |
| `agent1_type` / `agent2_type` | Agent types |
| `buyer_max` / `seller_min` | Private values |
| `zopa_low` / `zopa_high` | ZOPA bounds |
| `agreement` | Boolean |
| `final_price` | Agreed price (if any) |
| `agent1_utility` / `agent2_utility` | Final utilities |
| `turns` | Number of turns |
| `history_str` | Offer history |

---

## 🔗 Related Projects

- **Multi-Agent LLM Negotiation Research Domain**: Multi-issue allocation negotiations
- **OllamaHelperModule**: Ollama LLM integration utilities

---

## 📄 License

MIT License
