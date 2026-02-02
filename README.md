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
4. **Janus (HyperLoRA) Agents**: Trained agents with controllable behavior via ρ (rho) parameter

---

## 📂 Project Structure

```
Single Price Point Negotiation Domain/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── negotiation.py               # Main entry point for negotiation sessions
├── run_janus_vs_deterministic.py    # Janus vs deterministic strategy benchmarks
├── visualize_concessions.py     # Strategy visualization tool
│
├── config/
│   ├── settings.py              # Global configuration
│   └── *.txt                    # System instruction files
│
├── src/
│   ├── agents/                  # Agent implementations
│   │   ├── base_agent.py
│   │   ├── agent_factory.py
│   │   ├── price_strategies.py      # Deterministic strategies
│   │   ├── price_strategy_agent.py  # LLM wrapper for strategies
│   │   ├── basic_price_agent.py     # Free-form LLM agent
│   │   └── janus_agent.py           # HyperLoRA agent
│   │
│   ├── core/
│   │   └── price_structures.py      # PriceState, PriceAction dataclasses
│   │
│   ├── domain/
│   │   └── single_issue_price_domain.py
│   │
│   ├── training/                # Janus/HyperLoRA training pipeline
│   │   ├── hyper_lora.py
│   │   └── train_janus_hyperlora.py
│   │
│   ├── data_prep/
│   │   └── prepare_data.py      # JSONL to Parquet conversion
│   │
│   ├── logging/
│   │   ├── csv_logger.py
│   │   └── dataset_writer.py
│   │
│   └── utils/
│       └── __init__.py
│
├── datasets/                    # Generated datasets
│   ├── price_domain.jsonl
│   └── processed_tables/
│       └── decision_steps.parquet
│
├── checkpoints/                 # Trained model checkpoints
│   └── janus_v1/
│       └── final/
│
├── logs/                        # Experiment logs (CSV)
│
├── concession_plots/            # Visualization outputs
│
├── docs/
│   ├── hyperlora_training_report.md
│   └── price_strategies.md
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
python negotiation.py --agent1 basic --agent2 boulware_linear --rounds 5 --model qwen2:7b
```

**Deterministic vs Deterministic (No LLM):**
```bash
python negotiation.py --agent1 boulware_conceding --agent2 hardliner --rounds 10 --no-llm
```

### 3. Generate Training Dataset

```bash
python negotiation.py --dataset_mode --num_episodes 10000 --max_turns 20 --dataset_out datasets/price_domain.jsonl
```

### 4. Train Janus (HyperLoRA)

```bash
# Step 1: Prepare data
python src/data_prep/prepare_data.py --jsonl_path datasets/price_domain.jsonl --output_dir datasets/processed_tables

# Step 2: Train
python src/training/train_janus_hyperlora.py \
  --decision_steps_path datasets/processed_tables/decision_steps.parquet \
  --output_dir checkpoints/janus_v1 \
  --model_name "Qwen/Qwen2-7B-Instruct" \
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

Janus uses **HyperLoRA** to enable continuous behavioral control:

$$W = W_{\text{base}} + \frac{\alpha}{r} B \cdot \text{diag}(g(\rho)) \cdot A$$

Where:
- $\rho \in [0, 1]$: Control variable (0 = aggressive buyer, 1 = aggressive seller)
- $\rho = -1.0$: Impasse/failure mode
- $g(\rho)$: Gating vector from hypernetwork MLP

See [docs/hyperlora_training_report.md](docs/hyperlora_training_report.md) for full details.

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
