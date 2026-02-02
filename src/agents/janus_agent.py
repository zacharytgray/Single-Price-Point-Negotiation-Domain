"""
Janus Agent - HyperLoRA-controlled negotiation agent.
"""

import torch
import re
import os
import json
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.base_agent import BaseAgent
from src.training.hyper_lora import inject_hyperlora
from src.core.price_structures import PriceAction

# Global Cache for Janus Model
_JANUS_MODEL_CACHE = {}


class JanusAgent(BaseAgent):
    """
    Janus Agent: A HyperLoRA-controlled negotiation agent.
    
    This agent uses a single base LLM (Qwen2) equipped with HyperLoRA adapters.
    Behavior is controlled by a scalar 'rho' (0.0 to 1.0) injected at runtime.
    
    rho=0.0 -> Aggressive Buyer / Passive Seller
    rho=1.0 -> Aggressive Seller / Passive Buyer
    rho=-1.0 -> Forced Failure / Impasse Mode
    """
    
    def __init__(self, agent_id: int, role: str, model_path: str = "Qwen/Qwen2-7B-Instruct", 
                 adapter_path: str = "checkpoints/janus_v1/final", 
                 rho: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 p_low: float = 200.0,
                 p_high: float = 1500.0):
        
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
        except:
            print(f"[{role}] Warning: Could not load tokenizer from {adapter_path}, using {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.adapter_config = json.load(f)
        else:
            print(f"[{role}] Warning: No adapter_config.json found. Using defaults.")
            self.adapter_config = {
                "rank": 16, "alpha": 32, "hyper_hidden": 64,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }

        # Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # Inject HyperLoRA
        target_modules = self.adapter_config.get("target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        self.model = inject_hyperlora(
            self.model,
            target_module_names=target_modules,
            rank=self.adapter_config.get("rank", 16),
            alpha=self.adapter_config.get("alpha", 32),
            hyper_hidden=self.adapter_config.get("hyper_hidden", 64),
            dropout=0.0,
            use_fourier=self.adapter_config.get("use_fourier", False),
            fourier_freqs=self.adapter_config.get("fourier_freqs", 8),
            include_raw=self.adapter_config.get("include_raw", True)
        )

        # Load Weights
        weights_path = os.path.join(adapter_path, "adapter_state.pt")
        if os.path.exists(weights_path):
            print(f"[{role}] Loading HyperLoRA weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                critical_missing = [k for k in missing if "lora_" in k or "hyper_net.net" in k]
                if critical_missing:
                    print(f"[{role}] CRITICAL WARNING: Missing LoRA/HyperNet keys! ({len(critical_missing)})")
        else:
            raise FileNotFoundError(f"Could not find adapter weights at {weights_path}")

        self.model.eval()
        
        self.model._janus_adapter_config = self.adapter_config
        _JANUS_MODEL_CACHE[cache_key] = (self.model, self.tokenizer)
        print(f"[{role.upper()}] Model cached for {cache_key}")

    def _normalize(self, val: float, low: float, high: float) -> float:
        rng = high - low
        if rng < 1e-6:
            rng = 1.0
        return (val - low) / rng

    def _denormalize(self, norm_val: float, low: float, high: float) -> float:
        return low + norm_val * (high - low)
    
    async def generate_response(self) -> str:
        """
        Generates a text response using the Janus model.
        """
        ctx = self.domain_private_context
        if not ctx:
            return "ACCEPT"
            
        role = self.role
        turn = ctx.get("turn", 1)
        max_turns = ctx.get("max_turns", 20)
        
        p_low = self.p_low
        p_high = self.p_high
        
        reservation = ctx.get("max_willingness_to_pay") if role == "buyer" else ctx.get("min_acceptable_price")
        res_norm = self._normalize(reservation, p_low, p_high)
        
        history_str = "EMPTY"
        raw_history = ctx.get("history", [])
        
        k = 8
        if raw_history:
            pairs = []
            for h_role, h_price in raw_history[-k:]:
                h_norm = self._normalize(h_price, p_low, p_high)
                pairs.append(f"{h_role}:{h_norm:.4f}")
            if pairs:
                history_str = " ".join(pairs)
        
        last_offer_norm = "NA"
        if raw_history:
            last_offer_price = raw_history[-1][1]
            last_offer_norm = f"{self._normalize(last_offer_price, p_low, p_high):.4f}"
            
        turns_remaining = max_turns - turn
        
        prompt = (
            f"<RHO> {self.rho:.2f}\n"
            f"<ROLE> {role}\n"
            f"<TURN> {turn} / {max_turns}\n"
            f"<TURNS_REMAINING> {turns_remaining}\n"
            f"<RESERVATION_NORM> {res_norm:.4f}\n"
            f"<LAST_OFFER_NORM> {last_offer_norm}\n"
            f"<HISTORY> {history_str}\n"
            f"<INSTRUCTION> Output exactly one of:\n"
            f"ACCEPT\n"
            f"OFFER <PRICE_NORM>\n"
            f"<OUTPUT>\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.current_rho = torch.tensor([[self.rho]], device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
            
        output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if "ACCEPT" in output_text:
            return "ACCEPT"
            
        match = re.search(r"OFFER\s+([0-9\.]+)", output_text)
        if match:
            try:
                val_norm = float(match.group(1))
                price_real = self._denormalize(val_norm, p_low, p_high)
                price_real = round(price_real, 2)
                return f"OFFER {price_real}"
            except:
                pass
                
        return f"OFFER {reservation}"
        
    def add_to_memory(self, role: str, content: str):
        pass
        
    def reset_memory(self):
        pass

    async def propose_action(self, state: Any) -> PriceAction:
        """
        Generate an action given the current price state.
        Compatible with negotiation.py engine.
        """
        # Map PriceState to Janus context
        self.domain_private_context = {
            "role": state.role,
            "turn": state.timestep,
            "max_turns": state.max_turns,
            "history": state.offer_history,
            "max_willingness_to_pay": state.effective_reservation_price if state.role == "buyer" else None,
            "min_acceptable_price": state.effective_reservation_price if state.role == "seller" else None,
        }
        
        # Generate text response
        response = await self.generate_response()
        
        # Parse response into PriceAction
        if response.strip() == "ACCEPT":
            return PriceAction(type="ACCEPT", price=None)
            
        # Parse "OFFER X.XX"
        parts = response.split()
        if len(parts) >= 2 and parts[0] == "OFFER":
            try:
                price = float(parts[1])
                return PriceAction(type="OFFER", price=price)
            except ValueError:
                pass
                
        # Fallback to reservation if parsing fails
        fallback_price = state.effective_reservation_price
        return PriceAction(type="OFFER", price=fallback_price)
