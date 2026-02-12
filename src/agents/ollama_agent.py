"""
Ollama LLM Agent module for price negotiation.
"""

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
)


class OllamaAgent:
    """
    LLM Agent wrapper using Ollama for text generation.
    """
    
    def __init__(self, model_name: str, instructions_filepath: str):
        self.model_name = model_name
        self.instructions_filepath = instructions_filepath
        self.memory = [self._import_instructions()]
        self.temperature = MODEL_TEMPERATURE
        self.response_timeout = RESPONSE_TIMEOUT
        self.model = ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=self.temperature
        )
        if DEBUG_MODE:
            print(
                f"[DEBUG][OllamaAgent] model={self.model_name} temp={self.temperature} "
                f"timeout={self.response_timeout} instructions={self.instructions_filepath}"
            )

    def _import_instructions(self) -> SystemMessage:
        """Load system instructions from file."""
        instructions = ""
        try:
            with open(self.instructions_filepath, 'r') as f:
                instructions = f.read()
        except FileNotFoundError:
            print(f"{Fore.RED}Instructions file not found: {self.instructions_filepath}{Fore.RESET}")
            instructions = "You are a negotiation agent."
        return SystemMessage(content=instructions)

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
        """Reset memory to only contain the original system instructions."""
        original_system_message = self._import_instructions()
        self.memory = [original_system_message]

    async def generate_response(self, input_text_role: str = None, input_text: str = None) -> str:
        """Generate a response using the LLM."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if input_text and input_text_role:
                    self.add_to_memory(input_text_role, input_text)

                if DEBUG_MODE:
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
                response = await chain.ainvoke({})

                self.add_to_memory('assistant', response.content)

                raw_content = response.content.strip()
                
                if has_thinking_blocks(raw_content):
                    cleaned_content = strip_thinking_blocks(raw_content)
                    return cleaned_content
                else:
                    return raw_content
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
