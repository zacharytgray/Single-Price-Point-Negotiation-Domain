"""
Train HyperLoRA Janus Negotiation Agent.

This script trains a Janus agent using HyperLoRA on the price negotiation dataset.
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
    BitsAndBytesConfig
)
from tqdm import tqdm

from src.training.hyper_lora import inject_hyperlora, HyperLoRALinear

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Special tokens for the training format
SPECIAL_TOKENS = [
    "<RHO>",              # Control parameter for strategy (normalized outcome)
    "<TURN>",             # Current turn number / max turns
    "<TURNS_REMAINING>",  # Number of turns left before deadline
    "<RESERVATION_NORM>", # Normalized reservation price (0-1)
    "<LAST_OFFER_NORM>",  # Normalized last offer received
    "<HISTORY>",          # Sequence of past offers (padded)
    "<INSTRUCTION>",      # Delimiter section for task instruction
    "<OUTPUT>",           # Delimiter for target response
    "<RHO_FAIL>"          # Token indicating negotiation failure/impasse
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperLoRA Janus Negotiation Agent")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B", help="HF Model ID")
    parser.add_argument("--decision_steps_path", type=str, required=True, help="Path to decision_steps.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--k_history", type=int, default=8, help="Number of history items to show")
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
    
    # Eval / Saving
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.include_failures.lower() == "false":
        args.include_failures = False
    else:
        args.include_failures = True
        
    return args


def normalize_price(val: float, low: float, high: float) -> float:
    """Normalize price to [0, 1] given range."""
    rng = high - low
    if rng < 1e-6:
        rng = 1.0
    return (val - low) / rng


class NegotiationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, k_history: int):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_history = k_history
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        rho = float(row['rho_train'])
        role = str(row['role'])
        turn = int(row['turn'])
        max_turns = int(row['max_turns'])
        turns_rem = int(row['turns_remaining'])
        
        p_low = float(row['price_low']) if not pd.isna(row['price_low']) else 0.0
        p_high = float(row['price_high']) if not pd.isna(row['price_high']) else 2000.0
        
        res_price = float(row['reservation_price']) if not pd.isna(row['reservation_price']) else p_low
        res_norm = normalize_price(res_price, p_low, p_high)
        
        last_offer_norm = "NA"
        if not pd.isna(row['last_offer_price']):
            lo_val = float(row['last_offer_price'])
            last_offer_norm = f"{normalize_price(lo_val, p_low, p_high):.4f}"
            
        hist_str = "EMPTY"
        h_roles = row['history_roles']
        h_prices = row['history_prices']
        
        if hasattr(h_roles, 'tolist'):
            h_roles = h_roles.tolist()
        if hasattr(h_prices, 'tolist'):
            h_prices = h_prices.tolist()
        
        if h_roles and len(h_roles) > 0:
            # Pad with empty offers if not enough history
            if len(h_roles) < self.k_history:
                # Calculate needed padding
                padding_needed = self.k_history - len(h_roles)
                # Pad with placeholder values (e.g., "PAD:0.0000")
                pairs = [f"PAD:0.0000"] * padding_needed
                
                # Add actual history
                for r, p in zip(h_roles, h_prices):
                    p_norm = normalize_price(p, p_low, p_high)
                    pairs.append(f"{r}:{p_norm:.4f}")
            else:
                # Take last K offers
                start_k = len(h_roles) - self.k_history
                pairs = []
                for r, p in zip(h_roles[start_k:], h_prices[start_k:]):
                    p_norm = normalize_price(p, p_low, p_high)
                    pairs.append(f"{r}:{p_norm:.4f}")
            
            hist_str = " ".join(pairs)
        else:
            # Fully padded history if empty
            hist_str = " ".join([f"PAD:0.0000"] * self.k_history)
            
        t_action = row['target_action']
        target_str = ""
        if t_action == "ACCEPT":
            target_str = "ACCEPT"
        else:
            t_price = float(row['target_price']) if not pd.isna(row['target_price']) else 0.0
            t_p_norm = normalize_price(t_price, p_low, p_high)
            target_str = f"OFFER {t_p_norm:.4f}"
            
        if abs(rho - (-1.0)) < 1e-6:
            rho_token_str = "<RHO_FAIL>"
        else:
            rho_token_str = f"{rho:.2f}"
            
        prompt = (
            f"<RHO> {rho_token_str}\n"
            f"<TURN> {turn} / {max_turns}\n"
            f"<TURNS_REMAINING> {turns_rem}\n"
            f"<RESERVATION_NORM> {res_norm:.4f}\n"
            f"<LAST_OFFER_NORM> {last_offer_norm}\n"
            f"<HISTORY> {hist_str}\n"
            f"<INSTRUCTION> Output exactly one of:\n"
            f"ACCEPT\n"
            f"OFFER <PRICE_NORM>\n"
            f"<OUTPUT>\n"
        )
        
        full_text = prompt + target_str + self.tokenizer.eos_token
        
        input_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
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
            "rho": torch.tensor([rho], dtype=torch.float32)
        }


class HyperLoRACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [x['attention_mask'] for x in batch]
        labels = [x['labels'] for x in batch]
        rhos = [x['rho'] for x in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        rhos_tensor = torch.stack(rhos)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "rho": rhos_tensor
        }


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
        "augmented_tokens": SPECIAL_TOKENS
    }
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
        
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved adapter to {output_dir}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info(f"Loading data from {args.decision_steps_path}...")
    df = pd.read_parquet(args.decision_steps_path)
    logger.info(f"Loaded {len(df)} rows.")
    
    if not args.include_failures:
        logger.info("Excluding failures (rho == -1.0)...")
        df = df[df['rho_train'] != -1.0]
        logger.info(f"Filtered to {len(df)} rows.")
        
    success_df = df[df['rho_train'] != -1.0]
    if not success_df.empty:
        rhos = success_df['rho_train']
        logger.info(f"Success Rho Stats: Min={rhos.min():.4f}, Max={rhos.max():.4f}, Mean={rhos.mean():.4f}")
    
    logger.info(f"Failures count: {len(df[df['rho_train'] == -1.0])}")

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_size = int(len(shuffled) * 0.01)
    if eval_size < 1:
        eval_size = 0
    
    train_df = shuffled.iloc[eval_size:]
    eval_df = shuffled.iloc[:eval_size]
    
    logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
    
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Added {num_added} special tokens.")
    
    train_dataset = NegotiationDataset(train_df, tokenizer, args.max_length, args.k_history)
    eval_dataset = NegotiationDataset(eval_df, tokenizer, args.max_length, args.k_history)
    
    collator = HyperLoRACollator(tokenizer.pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if eval_size > 0 else None
    
    logger.info("Loading Base Model...")
    
    device_map = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info("Injecting HyperLoRA Modules...")
    model = inject_hyperlora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        hyper_hidden=args.hyper_hidden
    )
    
    if not args.use_qlora:
        model.to("cuda")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    num_training_steps = args.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )
    
    logger.info("Starting Training...")
    global_step = 0
    model.train()
    
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    epoch = 0
    
    while global_step < num_training_steps:
        epoch += 1
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            rhos = batch['rho'].to(model.device)
            
            model.current_rho = rhos
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if torch.isnan(loss).any():
                    logger.error(f"NaN loss detected at step {global_step}! Stopping.")
                    return

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item() * args.grad_accum)
                
                if global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_hyperlora_adapter(model, save_path, tokenizer, args)
                    
                if global_step >= num_training_steps:
                    break

    logger.info("Training Complete.")
    save_hyperlora_adapter(model, os.path.join(args.output_dir, "final"), tokenizer, args)


if __name__ == "__main__":
    train()
