"""
Janus Agent - HyperLoRA-controlled negotiation agent.

Inference prompt format matches training exactly (canonical format).
No <ROLE> token is used -- rho alone modulates behaviour.
"""

import torch
import re
import os
import json
from typing import Optional, Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.base_agent import BaseAgent
from src.training.hyper_lora import inject_hyperlora
from src.core.price_structures import PriceAction

# Re-use canonical helpers from training to guarantee identical prompts
from src.training.train_janus_hyperlora import (
    normalize_price,
    build_history_str,
    build_prompt,
    SPECIAL_TOKENS,
)

# Global Cache for Janus Model
_JANUS_MODEL_CACHE = {}

# Default history window size (overridden by adapter_config if saved)
_DEFAULT_K_HISTORY = 8


class JanusAgent(BaseAgent):
    """
    Janus Agent: A HyperLoRA-controlled negotiation agent.

    This agent uses a single base LLM equipped with HyperLoRA adapters.
    Behavior is controlled by a scalar 'rho' (0.0 to 1.0) injected at runtime.
    The user provides ONLY rho -- no explicit role token is sent to the model.

    rho in [0, 1] -> success-mode conditioning
    rho = -1.0    -> failure/impasse mode
    """

    def __init__(
        self,
        agent_id: int,
        role: str,
        model_path: str = "Qwen/Qwen2-7B-Instruct",
        adapter_path: str = "checkpoints/janus_v1/final",
        rho: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        p_low: float = 200.0,
        p_high: float = 1500.0,
    ):
        super().__init__(agent_id, f"janus_{rho}", "none")
        self.role = role
        self.rho = rho
        self.device = device
        self.adapter_path = adapter_path
        self.p_low = p_low
        self.p_high = p_high
        self.model_path = model_path

        print(f"[{role.upper()}] Initializing JanusAgent (rho={rho}) on {device}...")

        cache_key = f"{model_path}_{adapter_path}"

        if cache_key in _JANUS_MODEL_CACHE:
            print(f"[{role.upper()}] Using Cached Model for {cache_key}")
            self.model, self.tokenizer = _JANUS_MODEL_CACHE[cache_key]
            self.adapter_config = getattr(self.model, "_janus_adapter_config", {})
            return

        # Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            print(f"[{role}] Warning: Could not load tokenizer from {adapter_path}, using {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.adapter_config = json.load(f)
        else:
            print(f"[{role}] Warning: No adapter_config.json found. Using defaults.")
            self.adapter_config = {
                "rank": 16, "alpha": 32, "hyper_hidden": 64,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }

        # Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        # Inject HyperLoRA
        target_modules = self.adapter_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        self.model = inject_hyperlora(
            self.model,
            target_module_names=target_modules,
            rank=self.adapter_config.get("rank", 16),
            alpha=self.adapter_config.get("alpha", 32),
            hyper_hidden=self.adapter_config.get("hyper_hidden", 64),
            dropout=0.0,
            use_fourier=self.adapter_config.get("use_fourier", False),
            fourier_freqs=self.adapter_config.get("fourier_freqs", 8),
            include_raw=self.adapter_config.get("include_raw", True),
        )

        # Load Weights
        weights_path = os.path.join(adapter_path, "adapter_state.pt")
        if os.path.exists(weights_path):
            print(f"[{role}] Loading HyperLoRA weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                critical_missing = [k for k in missing if "lora_" in k or "hyper_net.net" in k]
                if critical_missing:
                    print(f"[{role}] CRITICAL WARNING: Missing LoRA/HyperNet keys! ({len(critical_missing)})")
        else:
            raise FileNotFoundError(f"Could not find adapter weights at {weights_path}")

        self.model.eval()

        self.model._janus_adapter_config = self.adapter_config
        _JANUS_MODEL_CACHE[cache_key] = (self.model, self.tokenizer)
        print(f"[{role.upper()}] Model cached for {cache_key}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _denormalize(self, norm_val: float, low: float, high: float) -> float:
        return low + norm_val * (high - low)

    def _get_k_history(self) -> int:
        """Return K from adapter config, falling back to default."""
        return self.adapter_config.get("k_history", _DEFAULT_K_HISTORY)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    async def generate_response(self) -> str:
        """Generates a text response using the Janus model.

        Builds the prompt using the SAME canonical ``build_prompt`` that
        training uses -- guaranteeing identical formatting.
        """
        ctx = self.domain_private_context
        if not ctx:
            return "ACCEPT"

        role = self.role
        turn = ctx.get("turn", 1)
        max_turns = ctx.get("max_turns", 20)

        p_low = self.p_low
        p_high = self.p_high

        reservation = (
            ctx.get("max_willingness_to_pay")
            if role == "buyer"
            else ctx.get("min_acceptable_price")
        )
        res_norm = normalize_price(reservation, p_low, p_high)

        # --- history -------------------------------------------------------
        raw_history: List[Tuple[str, float]] = ctx.get("history", [])
        k = self._get_k_history()

        h_roles = [h[0] for h in raw_history]
        h_prices = [float(h[1]) for h in raw_history]
        
        history_str, history_len = build_history_str(h_roles, h_prices, p_low, p_high, k)

        # --- last offer norm -----------------------------------------------
        last_offer_norm_str = "NA"
        if raw_history:
            last_offer_price = raw_history[-1][1]
            last_offer_norm_str = f"{normalize_price(last_offer_price, p_low, p_high):.4f}"

        turns_remaining = max_turns - turn

        # --- build prompt (identical to training) --------------------------
        rho_numeric = self.rho  # single source of truth

        prompt = build_prompt(
            rho=rho_numeric,
            turn=turn,
            max_turns=max_turns,
            turns_remaining=turns_remaining,
            reservation_norm=res_norm,
            last_offer_norm_str=last_offer_norm_str,
            history_len=history_len,
            history_str=history_str,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Set numeric rho on model (same value used for textual rho above)
        self.model.current_rho = torch.tensor(
            [[rho_numeric]], device=self.device, dtype=torch.float32
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        if "ACCEPT" in output_text:
            return "ACCEPT"

        match = re.search(r"OFFER\s+([0-9.]+)", output_text)
        if match:
            try:
                val_norm = float(match.group(1))
                price_real = self._denormalize(val_norm, p_low, p_high)
                price_real = round(price_real, 2)
                return f"OFFER {price_real}"
            except Exception:
                pass

        return f"OFFER {reservation}"

    # ------------------------------------------------------------------
    # interface stubs
    # ------------------------------------------------------------------
    def add_to_memory(self, role: str, content: str):
        pass

    def reset_memory(self):
        pass

    async def propose_action(self, state: Any) -> PriceAction:
        """Generate an action given the current price state.

        Compatible with negotiation.py engine.
        """
        self.domain_private_context = {
            "role": state.role,
            "turn": state.timestep,
            "max_turns": state.max_turns,
            "history": state.offer_history,
            "max_willingness_to_pay": state.effective_reservation_price if state.role == "buyer" else None,
            "min_acceptable_price": state.effective_reservation_price if state.role == "seller" else None,
        }

        response = await self.generate_response()

        if response.strip() == "ACCEPT":
            return PriceAction(type="ACCEPT", price=None)

        parts = response.split()
        if len(parts) >= 2 and parts[0] == "OFFER":
            try:
                price = float(parts[1])
                return PriceAction(type="OFFER", price=price)
            except ValueError:
                pass

        fallback_price = state.effective_reservation_price
        return PriceAction(type="OFFER", price=fallback_price)
