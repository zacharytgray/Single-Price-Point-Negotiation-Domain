"""
Ollama LLM Agent module for price negotiation.
"""

import re
import time
from typing import Optional, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from colorama import Fore
from src.utils.thinking_model_processor import strip_thinking_blocks, has_thinking_blocks
from config.settings import (
    DEBUG_MODE,
    MODEL_TEMPERATURE,
    RESPONSE_TIMEOUT,
    OLLAMA_BASE_URL,
    PRICE_RANGE_LOW,
    PRICE_RANGE_HIGH,
)


class OllamaAgent:
    """
    LLM Agent wrapper using Ollama for text generation.
    """
    
    def __init__(self, model_name: str, role: str, reservation_price: float,
                 max_turns: int = 20, system_instructions: str = None,
                 debug: bool = False):
        self.model_name = model_name
        self.role = role
        self.reservation_price = reservation_price
        self.max_turns = max_turns
        self.system_instructions = system_instructions
        self.debug = debug
        
        # Build memory - only add system instructions if provided (for Janus)
        self.memory = []
        if system_instructions:
            self.memory.append(SystemMessage(content=system_instructions))
            
        self.temperature = MODEL_TEMPERATURE
        self.response_timeout = RESPONSE_TIMEOUT
        self.model = ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=self.temperature
        )
        if DEBUG_MODE:
            print(
                f"[DEBUG][OllamaAgent] model={self.model_name} role={self.role} "
                f"reservation={self.reservation_price}"
            )

    def build_prompt(self, turn: int,
                     last_offer: Optional[float], history: List[Tuple[str, float]],
                     opponent_reservation: Optional[float] = None) -> str:
        """Build negotiation prompt for base model.

        Key design: frame the negotiation space so the model understands
        WHERE in the price range it should be operating, not just the
        reservation boundary.  Buyer wants price LOW, seller wants price HIGH.

        Args:
            opponent_reservation: The opponent's reservation price (buyer_max for seller,
                                  seller_min for buyer). Used to compute ZOPA-relative
                                  opening anchors so the seller doesn't anchor too low.
        """
        turns_remaining = self.max_turns - turn + 1

        if self.role == "buyer":
            # Buyer wants the LOWEST price possible.
            # Reservation = ceiling (absolute max they can pay).
            # Good opening = near the bottom of the public range.
            goal_text = "Your goal is to buy at the LOWEST price possible."
            reservation_text = (
                f"Your absolute maximum (walk-away price) is ${self.reservation_price:.2f}. "
                "You lose money on every dollar above the minimum the seller would accept, "
                "so you want to pay as far BELOW your maximum as you can."
            )
            # Buyer: start at 50-70% of reservation (aggressive but not extreme)
            opening_low = round(self.reservation_price * 0.5, 2)
            opening_high = round(self.reservation_price * 0.7, 2)
            strategy_block = (
                "STRATEGY:\n"
                f"- Open with a LOW offer (around ${opening_low:.2f}–${opening_high:.2f}) to anchor the negotiation.\n"
                "- Your offers should be LOWER than the seller's offers.\n"
                "- Increase your offer slowly over many turns — do NOT jump to a high price.\n"
                f"- NEVER offer more than ${self.reservation_price:.2f}.\n"
                "- ACCEPT if the seller's offer is low enough to be a good deal for you."
            )
            if last_offer is not None:
                situation = (
                    f"The seller's last offer was ${last_offer:.2f}. "
                    "If you're making a counteroffer instead of accepting, your counteroffer should be BELOW that price (you are trying to pull the price DOWN)."
                )
            else:
                situation = (
                    "You are making the opening offer. Start LOW — "
                    f"well below your maximum of ${self.reservation_price:.2f}."
                )
        else:
            # Seller wants the HIGHEST price possible.
            # Reservation = floor (absolute min they can accept).
            # Good opening = near the top of the public range.
            goal_text = "Your goal is to sell at the HIGHEST price possible."
            reservation_text = (
                f"Your absolute minimum (walk-away price) is ${self.reservation_price:.2f}. "
                "You lose money on every dollar below the maximum the buyer would pay, "
                "so you want to sell as far ABOVE your minimum as you can."
            )
            # Seller should anchor near the TOP of the ZOPA (near buyer_max), not
            # relative to their own floor — multiplying seller_min by 1.3x lands
            # well below the midpoint when the ZOPA is wide.
            if opponent_reservation is not None:
                # opponent_reservation is buyer_max; anchor at 80-95% of ZOPA
                zopa = opponent_reservation - self.reservation_price
                opening_low  = round(self.reservation_price + zopa * 0.80, 2)
                opening_high = round(self.reservation_price + zopa * 0.95, 2)
            else:
                # Fallback: 130-150% of reservation (legacy behaviour)
                opening_low  = round(self.reservation_price * 1.3, 2)
                opening_high = round(self.reservation_price * 1.5, 2)
            strategy_block = (
                "STRATEGY:\n"
                f"- Open with a HIGH offer (around ${opening_low:.2f}–${opening_high:.2f}) to anchor the negotiation.\n"
                "- Your offers should be HIGHER than the buyer's offers.\n"
                "- Decrease your offer slowly over many turns — do NOT drop to a low price.\n"
                f"- NEVER offer less than ${self.reservation_price:.2f}.\n"
                "- ACCEPT if the buyer's offer is high enough to be a good deal for you."
            )
            if last_offer is not None:
                situation = (
                    f"The buyer's last offer was ${last_offer:.2f}. "
                    "If you're making a counteroffer instead of accepting, your counteroffer should be ABOVE that price (you are trying to push the price UP)."
                )
            else:
                situation = (
                    "You are making the opening offer. Start HIGH — "
                    f"well above your minimum of ${self.reservation_price:.2f}."
                )

        # Time pressure warning escalates as turns run out
        if turns_remaining <= 3:
            time_warning = (
                f"\n⚠️  FINAL WARNING: Only {turns_remaining} turns left. "
                "If you do not reach agreement soon, the negotiation ends with NO DEAL. "
                "You will receive your reservation price (your worst possible outcome). "
                "Any agreement at all is better than no agreement."
            )
        elif turns_remaining <= 5:
            time_warning = (
                f"\n⚠️  TIME PRESSURE: Only {turns_remaining} turns remaining. "
                "The risk of NO DEAL is increasing. Consider whether holding out is worth the risk."
                "Any agreement at all is better than no agreement."
            )
        else:
            time_warning = ""

        prompt = f"""You are the {self.role.upper()} in a single-issue price negotiation.
{goal_text}

RESERVATION PRICE: ${self.reservation_price:.2f}
{reservation_text}

NEGOTIATION CONTEXT:
- Turn {turn} of {self.max_turns} ({turns_remaining} remaining)
- Trading range: ${PRICE_RANGE_LOW:.2f} – ${PRICE_RANGE_HIGH:.2f}

CRITICAL: If no agreement is reached by turn {self.max_turns}, the negotiation ends with NO DEAL.
NO DEAL means that you will receive your reservation price, which is the worst possible outcome for you.
Any agreement (even a bad one) is better than no agreement.
{time_warning}
"""

        if last_offer is not None:
            prompt += f"\nOPPONENT'S LAST OFFER: ${last_offer:.2f}\n"

        if history:
            prompt += "\nOFFER HISTORY:\n"
            for i, (h_role, price) in enumerate(history[-8:], 1):
                prompt += f"  {i}. {h_role.upper()}: ${price:.2f}\n"

        prompt += f"""
{strategy_block}

CURRENT SITUATION: {situation}

Respond with EXACTLY one of:
  OFFER $X.XX
  ACCEPT"""

        return prompt

    def _normalize_model_response(self, raw_content: str) -> str:
        """Normalize model output to strict benchmark action format."""
        if not raw_content:
            return ""

        text = raw_content.strip()
        text_upper = text.upper()

        if "ACCEPT" in text_upper and "OFFER" not in text_upper:
            return "ACCEPT"

        offer_match = re.search(r"OFFER\s+\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if offer_match:
            try:
                price = float(offer_match.group(1))
                if self.role == "buyer":
                    price = min(price, self.reservation_price)
                else:
                    price = max(price, self.reservation_price)
                return f"OFFER ${price:.2f}"
            except Exception:
                pass

        dollar_match = re.search(r"\$\s*(-?\d+(?:\.\d+)?)", text)
        if dollar_match:
            try:
                price = float(dollar_match.group(1))
                if self.role == "buyer":
                    price = min(price, self.reservation_price)
                else:
                    price = max(price, self.reservation_price)
                return f"OFFER ${price:.2f}"
            except Exception:
                pass

        return text

    def add_to_memory(self, role: str, content: str):
        """Add a message to conversation memory."""
        if role == 'system':
            self.memory.append(SystemMessage(content=content))
        elif role == 'user':
            self.memory.append(HumanMessage(content=content))
        elif role == 'assistant':
            if isinstance(content, AIMessage):
                self.memory.append(content)
            else:
                self.memory.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")

        if DEBUG_MODE:
            snippet = content if isinstance(content, str) else str(content)
            snippet = snippet.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            print(f"[DEBUG][OllamaAgent] add_to_memory role={role} content=\"{snippet}\"")
    
    def reset_memory(self):
        """Reset memory to only contain the original system instructions if any."""
        self.memory = []
        if self.system_instructions:
            self.memory.append(SystemMessage(content=self.system_instructions))

    async def generate_response(self, input_text_role: str = None, input_text: str = None) -> str:
        """Generate a response using the LLM.
        
        For base model: each turn is stateless (no message history accumulation).
        The prompt already contains embedded offer history, so we reset memory each turn.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Reset memory before each turn (stateless prompting)
                # Offer history is already embedded in the prompt text
                self.reset_memory()
                
                if input_text and input_text_role:
                    self.add_to_memory(input_text_role, input_text)

                if DEBUG_MODE or self.debug:
                    tail_msgs = self.memory[-4:] if len(self.memory) >= 4 else self.memory
                    print("[DEBUG][OllamaAgent] context_tail=")
                    for msg in tail_msgs:
                        role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
                        text = msg.content.replace("\n", " ")
                        if len(text) > 240:
                            text = text[:240] + "..."
                        print(f"  - {role}: {text}")

                history = ChatPromptTemplate.from_messages(self.memory)
                chain = history | self.model
                t0 = time.perf_counter()
                response = await chain.ainvoke({})
                latency_s = time.perf_counter() - t0

                # Do NOT add response to memory (stateless)

                raw_content = response.content.strip()
                
                if has_thinking_blocks(raw_content):
                    cleaned_content = strip_thinking_blocks(raw_content)
                else:
                    cleaned_content = raw_content

                normalized = self._normalize_model_response(cleaned_content)

                if DEBUG_MODE or self.debug:
                    print(f"[DEBUG][OllamaAgent] latency={latency_s:.3f}s")
                    print(f"[DEBUG][OllamaAgent] raw_response={raw_content}")
                    print(f"[DEBUG][OllamaAgent] normalized_response={normalized}")

                # Memory already reset at start of this method (stateless)
                return normalized
            except Exception as e:
                print(f"{Fore.RED}Error generating response (attempt {attempt}): {e}{Fore.RESET}")
                if attempt == max_retries:
                    return ""

    def print_memory(self, skip_system_message: bool = False):
        """Print conversation history for debugging."""
        if skip_system_message:
            messages_to_print = [msg for msg in self.memory if not isinstance(msg, SystemMessage)]
        else:
            messages_to_print = self.memory
        print(f"----------------{Fore.LIGHTYELLOW_EX}Conversation History:{Fore.RESET}----------------")
        for message in messages_to_print:
            if isinstance(message, SystemMessage):
                print(f"{Fore.LIGHTRED_EX}System: {message.content}{Fore.RESET}")
            elif isinstance(message, HumanMessage):
                print(f"{Fore.LIGHTGREEN_EX}User: {message.content}{Fore.RESET}")
            elif isinstance(message, AIMessage):
                print(f"{Fore.LIGHTBLUE_EX}Agent: {message.content}{Fore.RESET}")
            else:
                print(f"Unknown message type: {message}")
            print("----------------------------------------------------------------------------------------")
        print(f"----------------{Fore.LIGHTYELLOW_EX}END History:{Fore.RESET}----------------")
