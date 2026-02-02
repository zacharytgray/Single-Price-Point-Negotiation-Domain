# HyperLoRA (Janus) Training & Inference Report

This report documents the full HyperLoRA training and inference pipeline, including the model architecture, data flow, training instance format, and runtime usage.

---

## 1. Conceptual Overview: What HyperLoRA Is

HyperLoRA is a conditional LoRA system. It keeps a base LLM frozen and adds a low‑rank update to select linear layers, but *modulates* that low‑rank update with a hypernetwork conditioned on a scalar control variable $\rho$.

**Core equation (per linear layer):**

$$
W = W_{\text{base}} + \frac{\alpha}{r} \; B\;\mathrm{diag}(g(\rho))\;A
$$

Where:
- $W_{\text{base}}$ is the frozen weight matrix of the original linear layer.
- $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$ are LoRA matrices.
- $r$ is the LoRA rank.
- $\alpha$ is the LoRA scaling factor.
- $g(\rho) \in \mathbb{R}^{r}$ is a **gating vector** produced by a hypernetwork, which controls how much each rank component contributes.

This lets a *single* trained model exhibit a continuum of behaviors by changing $\rho$ at inference time (e.g., aggressive vs. conceding negotiation styles).

### 1.1 Rho ($\rho$) Semantics
The system uses $\rho$ as a continuous control signal:
- $\rho \in [0, 1]$ maps to a normalized outcome within the buyer–seller bargaining range.
- $\rho = -1.0$ denotes an "impasse/failure" mode in data labeling.

---

## 2. HyperLoRA Architecture (Code-Level)

### 2.1 Hypernetwork: `RhoHyperNet`
Defined in [src/training/hyper_lora.py](../src/training/hyper_lora.py).

**Purpose:** Map scalar $\rho$ to a gating vector $g(\rho)$ of size `rank`.

**Structure:**
- Input: $\rho$ of shape `[B, 1]`.
- Optional Fourier features (sin/cos) for periodic encoding.
- MLP: Linear → SiLU → Linear → SiLU → Linear.
- Final activation: Sigmoid (default).

### 2.2 HyperLoRA Linear Wrapper: `HyperLoRALinear`
Each target `nn.Linear` is replaced with a wrapper that:
1. Runs the frozen base layer to get $y_{\text{base}}$.
2. Fetches current $\rho$ using a getter closure bound to `model.current_rho`.
3. Computes $g(\rho)$ via the hypernetwork.
4. Applies LoRA using $A$ and $B$ matrices, gating by $g(\rho)$.
5. Adds the scaled LoRA delta to the base output.

### 2.3 Module Injection: `inject_hyperlora`
- Adds `model.current_rho` to hold the batch's $\rho$ values.
- Finds target `nn.Linear` modules by suffix (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Replaces those modules with `HyperLoRALinear` wrappers.
- Freezes everything except LoRA and hypernetwork parameters.

---

## 3. Dataset Generation Flow

### 3.1 Negotiation Episodes → JSONL
Dataset generation runs through [negotiation.py](../negotiation.py) or via the `DatasetWriter` class.

#### Rho Calculation
- $\rho$ is computed **only if an agreement is reached**:
  $$\rho = \frac{\text{FinalPrice} - \text{SellerMin}}{\text{BuyerMax} - \text{SellerMin}}$$
- The same $\rho$ is written into **every step record** of that trajectory.
- If no agreement is reached, the trajectory can optionally be saved with `rho = -1.0`.

### 3.2 JSONL → Parquet
The transformation happens in [src/data_prep/prepare_data.py](../src/data_prep/prepare_data.py).

Key steps:
1. Group JSONL entries by `trajectory_id`.
2. Determine success (any `ACCEPT`) or failure (turns == max_turns).
3. Assign training $\rho$:
   - Success → `rho_outcome` (from JSONL).
   - Failure → `rho_train = -1.0`.
4. For each step, build a row with all needed training features.

---

## 4. Training Pipeline

Training is implemented in [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py).

### 4.1 Special Tokens
```
<RHO> <TURN> <TURNS_REMAINING> <RESERVATION_NORM> <LAST_OFFER_NORM> <HISTORY> <INSTRUCTION> <OUTPUT> <RHO_FAIL>
```

### 4.2 Training Instance Format
```
<RHO> 0.53
<TURN> 7 / 20
<TURNS_REMAINING> 13
<RESERVATION_NORM> 0.8421
<LAST_OFFER_NORM> 0.4123
<HISTORY> PAD:0.0000 PAD:0.0000 ... buyer:0.3000 seller:0.4000 buyer:0.4200
<INSTRUCTION> Output exactly one of:
ACCEPT
OFFER <PRICE_NORM>
<OUTPUT>
```
*Note: `<ROLE>` is omitted as the model learns behavior from `<RHO>`. History is padded to fixed length `k` with `PAD:0.0000`.*

The target is appended immediately after `<OUTPUT>`.

### 4.3 Key Training Steps
1. **Set `model.current_rho`** from the batch.
2. **Forward Pass** with standard causal LM loss.
3. **Labels masked** so loss only applies to target tokens.

### 4.4 Checkpoint Saving
`save_hyperlora_adapter` writes:
- `adapter_state.pt`: only trainable HyperLoRA parameters.
- `adapter_config.json`: configuration.
- Tokenizer files.

---

## 5. Inference & Runtime Usage

### 5.1 Loading the Trained Adapter
1. Load tokenizer from adapter path.
2. Read `adapter_config.json`.
3. Load base model.
4. Inject HyperLoRA modules.
5. Load `adapter_state.pt`.

### 5.2 Runtime Usage
Before generation:
- Set `model.current_rho` to desired value.
- Build prompt with same structure as training.
- Generate continuation.
- Parse output as `ACCEPT` or `OFFER <PRICE>`.

### 5.3 Key Contract
The model expects **two synchronized signals**:
1. A **textual rho** in the prompt.
2. A **numeric rho** tensor in `model.current_rho`.

---

## 6. End‑to‑End Flow

1. **Generate dataset** (deterministic strategies): [negotiation.py](../negotiation.py)
2. **Write JSONL + Rho**: [src/logging/dataset_writer.py](../src/logging/dataset_writer.py)
3. **Prepare Parquet**: [src/data_prep/prepare_data.py](../src/data_prep/prepare_data.py)
4. **Train HyperLoRA**: [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py)
5. **Save Adapter**: `adapter_state.pt` + `adapter_config.json`
6. **Load & Run Janus Agent**: [src/agents/janus_agent.py](../src/agents/janus_agent.py)

---

## 7. Files Referenced

- [src/training/hyper_lora.py](../src/training/hyper_lora.py)
- [src/training/train_janus_hyperlora.py](../src/training/train_janus_hyperlora.py)
- [src/agents/janus_agent.py](../src/agents/janus_agent.py)
- [src/data_prep/prepare_data.py](../src/data_prep/prepare_data.py)
- [src/logging/dataset_writer.py](../src/logging/dataset_writer.py)
