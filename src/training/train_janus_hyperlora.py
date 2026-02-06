"""
Train HyperLoRA Janus Negotiation Agent.

This script trains a Janus agent using HyperLoRA on the price negotiation dataset.

Prompt contract (canonical format -- used identically at training AND inference):

    <RHO> {<RHO_FAIL> | value:.4f}
    <SUCCESS> {0|1}
    <TURN> {turn} / {max_turns}
    <TURNS_REMAINING> {turns_remaining}
    <RESERVATION_NORM> {reservation_norm:.4f}
    <LAST_OFFER_NORM> {last_offer_norm:.4f | NA}
    <HISTORY_LEN> {n}
    <HISTORY> slot1 slot2 ... slotK
    <INSTRUCTION> Output exactly one of:
    ACCEPT
    OFFER <PRICE_NORM>
    <OUTPUT>

Target: "ACCEPT" or "OFFER {price_norm:.4f}"

No <ROLE> token is ever included.  Rho alone modulates behaviour.
"""

import os
import sys
import argparse
import random
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from src.training.hyper_lora import inject_hyperlora, HyperLoRALinear

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens -- canonical list.  No <ROLE> token.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<RHO>",
    "<RHO_FAIL>",
    "<SUCCESS>",
    "<TURN>",
    "<TURNS_REMAINING>",
    "<RESERVATION_NORM>",
    "<LAST_OFFER_NORM>",
    "<HISTORY_LEN>",
    "<HISTORY>",
    "<INSTRUCTION>",
    "<OUTPUT>",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_price(val: float, low: float, high: float) -> float:
    """Normalize *val* to [0, 1] given range [low, high].

    Returns clamped result.  Callers should pre-filter rows where
    high - low <= 0 (logged as a warning inside the dataset).
    """
    denom = high - low
    if denom < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (val - low) / denom))


def format_rho_text(rho_value: float) -> str:
    """Single authoritative function: numeric rho -> textual token string.

    * rho == -1.0  ->  "<RHO> <RHO_FAIL>"
    * otherwise    ->  "<RHO> {rho:.4f}"

    This is the ONLY place where the textual rho representation is built.
    Both training and inference must call this.
    """
    if abs(rho_value - (-1.0)) < 1e-6:
        return "<RHO> <RHO_FAIL>"
    return f"<RHO> {rho_value:.4f}"


def format_success_text(rho_value: float) -> str:
    """Return '<SUCCESS> 0' for failures, '<SUCCESS> 1' otherwise."""
    if abs(rho_value - (-1.0)) < 1e-6:
        return "<SUCCESS> 0"
    return "<SUCCESS> 1"


def build_history_str(
    history_roles: List[str],
    history_prices: List[float],
    p_low: float,
    p_high: float,
    k: int,
) -> Tuple[str, int]:
    """Build the canonical fixed-K history string.

    Returns (history_text, num_real) where *history_text* contains
    exactly *k* space-separated slots and *num_real* counts real entries.

    Slots are ordered oldest -> newest within the window.
    Real slot:  "{side}:{price_norm:.4f}"
    Empty slot: "EMPTY:EMPTY"
    """
    if hasattr(history_roles, "tolist"):
        history_roles = history_roles.tolist()
    if hasattr(history_prices, "tolist"):
        history_prices = history_prices.tolist()

    if not history_roles:
        history_roles = []
    if not history_prices:
        history_prices = []

    n = len(history_roles)

    # Take last-K real entries
    if n > k:
        history_roles = history_roles[-k:]
        history_prices = history_prices[-k:]
        n = k

    # Build real pairs (oldest -> newest)
    pairs: List[str] = []
    for side, price in zip(history_roles, history_prices):
        p_norm = normalize_price(price, p_low, p_high)
        pairs.append(f"{side}:{p_norm:.4f}")

    num_real = len(pairs)

    # Pad with EMPTY:EMPTY at the front (oldest slots are empty)
    padding_needed = k - num_real
    slots = ["EMPTY:EMPTY"] * padding_needed + pairs

    return " ".join(slots), num_real


def build_prompt(
    rho: float,
    turn: int,
    max_turns: int,
    turns_remaining: int,
    reservation_norm: float,
    last_offer_norm_str: str,
    history_len: int,
    history_str: str,
) -> str:
    """Build the canonical prompt string.

    This is the SINGLE source of truth for prompt layout.
    Used identically at training time and inference time.
    """
    rho_line = format_rho_text(rho)
    success_line = format_success_text(rho)

    prompt = (
        f"{rho_line}\n"
        f"{success_line}\n"
        f"<TURN> {turn} / {max_turns}\n"
        f"<TURNS_REMAINING> {turns_remaining}\n"
        f"<RESERVATION_NORM> {reservation_norm:.4f}\n"
        f"<LAST_OFFER_NORM> {last_offer_norm_str}\n"
        f"<HISTORY_LEN> {history_len}\n"
        f"<HISTORY> {history_str}\n"
        f"<INSTRUCTION> Output exactly one of:\n"
        f"ACCEPT\n"
        f"OFFER <PRICE_NORM>\n"
        f"<OUTPUT>\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperLoRA Janus Negotiation Agent")

    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B", help="HF Model ID")
    parser.add_argument("--decision_steps_path", type=str, required=True, help="Path to decision_steps.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--k_history", type=int, default=8, help="Number of history slots (K)")
    parser.add_argument("--include_failures", type=str, default="true", help="Include rho=-1.0 trajectories?")

    # Training Hyperparams
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit loading (QLoRA)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # HyperLoRA Params
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hyper_hidden", type=int, default=64)

    # Regularization -- smoothness (existing, optional)
    parser.add_argument("--lambda_smooth", type=float, default=0.0,
                        help="Weight for gate-smoothness regulariser (0 = off) [RESERVED - NOT CURRENTLY IMPLEMENTED]")

    # Regularization -- gate separation (mismatch E)
    parser.add_argument("--lambda_sep", type=float, default=0.0,
                        help="Weight for gate-separation regulariser (0 = off)")
    parser.add_argument("--sep_exclude_failures", type=str, default="true",
                        help="Exclude rho=-1 rows from separation loss")

    # Eval / Saving
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)

    args = parser.parse_args()

    # Normalise boolean-ish strings
    args.include_failures = args.include_failures.lower() != "false"
    args.sep_exclude_failures = args.sep_exclude_failures.lower() != "false"

    return args


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NegotiationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, k_history: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_history = k_history

        # Pre-filter rows with degenerate price ranges
        good_mask = (df["price_high"] - df["price_low"]) > 1e-6
        n_dropped = int((~good_mask).sum())
        if n_dropped > 0:
            logger.warning("Dropped %d rows with degenerate price range (high - low <= 0).", n_dropped)
        self.data = df[good_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- scalars -------------------------------------------------------
        rho: float = float(row["rho_train"])
        turn: int = int(row["turn"])
        max_turns: int = int(row["max_turns"])
        turns_rem: int = int(row["turns_remaining"])

        p_low = float(row["price_low"]) if not pd.isna(row["price_low"]) else 0.0
        p_high = float(row["price_high"]) if not pd.isna(row["price_high"]) else 2000.0

        res_price = float(row["reservation_price"]) if not pd.isna(row["reservation_price"]) else p_low
        res_norm = normalize_price(res_price, p_low, p_high)

        # --- last offer norm -----------------------------------------------
        last_offer_norm_str = "NA"
        if not pd.isna(row["last_offer_price"]):
            lo_val = float(row["last_offer_price"])
            last_offer_norm_str = f"{normalize_price(lo_val, p_low, p_high):.4f}"

        # --- history (mismatch B) ------------------------------------------
        h_roles = row["history_roles"]
        h_prices = row["history_prices"]
        hist_str, hist_len = build_history_str(h_roles, h_prices, p_low, p_high, self.k_history)

        # --- target ---------------------------------------------------------
        t_action = row["target_action"]
        if t_action == "ACCEPT":
            target_str = "ACCEPT"
        else:
            t_price = float(row["target_price"]) if not pd.isna(row["target_price"]) else 0.0
            t_p_norm = normalize_price(t_price, p_low, p_high)
            target_str = f"OFFER {t_p_norm:.4f}"

        # --- prompt (mismatches A, C, D, F) ---------------------------------
        prompt = build_prompt(
            rho=rho,
            turn=turn,
            max_turns=max_turns,
            turns_remaining=turns_rem,
            reservation_norm=res_norm,
            last_offer_norm_str=last_offer_norm_str,
            history_len=hist_len,
            history_str=hist_str,
        )

        full_text = prompt + target_str + self.tokenizer.eos_token

        input_enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = input_enc.input_ids[0]
        attention_mask = input_enc.attention_mask[0]

        labels = input_ids.clone()

        prompt_enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]

        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rho": torch.tensor([rho], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class HyperLoRACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]
        rhos = [x["rho"] for x in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        rhos_tensor = torch.stack(rhos)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "rho": rhos_tensor,
        }


# ---------------------------------------------------------------------------
# Gate separation regulariser  (mismatch E)
# ---------------------------------------------------------------------------
def compute_gate_separation_loss(
    model: nn.Module,
    rho: torch.Tensor,
    exclude_failures: bool = True,
) -> torch.Tensor:
    """Encourage different rho values to produce different gate vectors.

    For a batch of rho values we:
    1. Build a random permutation rho_perm.
    2. Forward both through every HyperLoRALinear.hyper_net.
    3. D = mean(||g - g_perm||^2) across layers and batch.
    4. w = mean(clamp(|rho - rho_perm|, 0, 1))  -- target separation weight.
    5. loss = -D * w  (we minimise loss, so negative -> maximise distance).

    Failure rows (rho == -1.0) are excluded from the computation when
    *exclude_failures* is True.
    """
    device = rho.device
    rho_flat = rho.view(-1, 1)
    B = rho_flat.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if exclude_failures:
        mask = (rho_flat.squeeze(-1) > -0.99)
        valid_idx = mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        rho_flat = rho_flat[valid_idx]
        B = rho_flat.shape[0]

    perm = torch.randperm(B, device=device)
    rho_perm = rho_flat[perm]

    hyper_modules: List[HyperLoRALinear] = [
        m for m in model.modules() if isinstance(m, HyperLoRALinear)
    ]
    if not hyper_modules:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_d = torch.tensor(0.0, device=device)
    for hlm in hyper_modules:
        g = hlm.hyper_net(rho_flat)
        g_perm = hlm.hyper_net(rho_perm)
        total_d = total_d + ((g - g_perm) ** 2).mean()

    avg_d = total_d / len(hyper_modules)
    w = torch.clamp((rho_flat - rho_perm).abs(), 0.0, 1.0).mean()

    return -(avg_d * w)


# ---------------------------------------------------------------------------
# Save / utils
# ---------------------------------------------------------------------------
def save_hyperlora_adapter(model, output_dir, tokenizer, args):
    """Save only the trainable HyperLoRA parameters."""
    os.makedirs(output_dir, exist_ok=True)

    to_save = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            to_save[n] = p.cpu()

    torch.save(to_save, os.path.join(output_dir, "adapter_state.pt"))

    config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "hyper_hidden": args.hyper_hidden,
        "base_model": args.model_name,
        "k_history": args.k_history,
        "augmented_tokens": SPECIAL_TOKENS,
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    tokenizer.save_pretrained(output_dir)
    logger.info("Saved adapter to %s", output_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train():
    args = parse_args()
    set_seed(args.seed)

    # --- data --------------------------------------------------------------
    logger.info("Loading data from %s...", args.decision_steps_path)
    df = pd.read_parquet(args.decision_steps_path)
    logger.info("Loaded %d rows.", len(df))

    if not args.include_failures:
        logger.info("Excluding failures (rho == -1.0)...")
        df = df[df["rho_train"] != -1.0]
        logger.info("Filtered to %d rows.", len(df))

    success_df = df[df["rho_train"] != -1.0]
    if not success_df.empty:
        rhos = success_df["rho_train"]
        logger.info("Success Rho Stats: Min=%.4f, Max=%.4f, Mean=%.4f", rhos.min(), rhos.max(), rhos.mean())

    logger.info("Failures count: %d", len(df[df["rho_train"] == -1.0]))

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_size = max(int(len(shuffled) * 0.01), 0)
    if eval_size < 1:
        eval_size = 0

    train_df = shuffled.iloc[eval_size:]
    eval_df = shuffled.iloc[:eval_size]
    logger.info("Train size: %d, Eval size: %d", len(train_df), len(eval_df))

    # --- tokenizer ---------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Added %d special tokens.", num_added)

    # --- datasets ----------------------------------------------------------
    train_dataset = NegotiationDataset(train_df, tokenizer, args.max_length, args.k_history)
    eval_dataset = NegotiationDataset(eval_df, tokenizer, args.max_length, args.k_history)

    collator = HyperLoRACollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0
    )
    eval_loader = (
        DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if eval_size > 0
        else None
    )

    # --- model -------------------------------------------------------------
    logger.info("Loading Base Model...")

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )

    model.resize_token_embeddings(len(tokenizer))

    logger.info("Injecting HyperLoRA Modules...")
    model = inject_hyperlora(
        model, rank=args.rank, alpha=args.alpha, dropout=args.dropout, hyper_hidden=args.hyper_hidden
    )

    if not args.use_qlora:
        model.to("cuda")

    # --- optimiser ---------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # --- training loop -----------------------------------------------------
    logger.info("Starting Training...")
    global_step = 0
    model.train()
    progress_bar = tqdm(total=args.max_steps, desc="Training")
    epoch = 0

    while global_step < args.max_steps:
        epoch += 1
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            rhos = batch["rho"].to(model.device)

            model.current_rho = rhos

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs.loss / args.grad_accum

            total_loss = lm_loss

            # --- optional: gate separation loss (mismatch E) ---------------
            if args.lambda_sep > 0:
                sep_loss = compute_gate_separation_loss(
                    model, rhos, exclude_failures=args.sep_exclude_failures
                )
                total_loss = total_loss + (args.lambda_sep * sep_loss) / args.grad_accum

            total_loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if torch.isnan(lm_loss).any():
                    logger.error("NaN loss detected at step %d! Stopping.", global_step)
                    return

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=lm_loss.item() * args.grad_accum)

                if global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_hyperlora_adapter(model, save_path, tokenizer, args)

                if global_step >= args.max_steps:
                    break

    logger.info("Training Complete.")
    save_hyperlora_adapter(model, os.path.join(args.output_dir, "final"), tokenizer, args)


if __name__ == "__main__":
    train()
