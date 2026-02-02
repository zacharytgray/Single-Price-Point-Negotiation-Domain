"""
Configuration settings for the Single Price Point Negotiation Domain.
Centralizes all configurable parameters and constants.
"""

# =============================================================================
# MODEL AND AGENT SETTINGS
# =============================================================================

# Default model settings
DEFAULT_MODEL_NAME = "qwen2:7b"
MODEL_TEMPERATURE = 0.2
RESPONSE_TIMEOUT = 60
OLLAMA_BASE_URL = "http://localhost:11434"

# Agent configuration
PRICE_SYSTEM_INSTRUCTIONS_FILE = "config/price_system_instructions.txt"
PRICE_DETERMINISTIC_INSTRUCTIONS_FILE = "config/price_determ_instructions.txt"

# =============================================================================
# NEGOTIATION SETTINGS
# =============================================================================

# Round settings
DEFAULT_NUM_ROUNDS = 100
MAX_TURNS_PER_ROUND = 20
MAX_TURNS = MAX_TURNS_PER_ROUND  # Alias for compatibility
MAX_RETRIES_PER_INVALID_PROPOSAL = 3

# Price domain defaults (Paper Replication)
# Buyer Max ~ N(900, 50)
# Seller Min = Buyer Max - 500 (Fixed ZOPA width)
DEFAULT_BUYER_MAX_MEAN = 900.0
DEFAULT_BUYER_MAX_STD = 50.0
DEFAULT_ZOPA_WIDTH = 500.0
FIXED_ZOPA_WIDTH = DEFAULT_ZOPA_WIDTH  # Alias

# Public price range (visible to all agents)
PUBLIC_PRICE_RANGE_MIN = 200.0
PUBLIC_PRICE_RANGE_MAX = 1500.0
PRICE_RANGE_LOW = PUBLIC_PRICE_RANGE_MIN  # Alias
PRICE_RANGE_HIGH = PUBLIC_PRICE_RANGE_MAX  # Alias

# Agreement detection
AGREEMENT_KEYWORDS = ["ACCEPT", "AGREE"]
OFFER_KEYWORD = "OFFER"

# Janus defaults
JANUS_ADAPTER_PATH = "checkpoints/final"
JANUS_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"

# =============================================================================
# LOGGING AND OUTPUT SETTINGS
# =============================================================================

# CSV logging
DEFAULT_LOG_DIR = "logs"
DEFAULT_RESULTS_DIR = "results"
CSV_ENCODING = "utf-8"
CSV_DATE_FORMAT = "%Y%m%d"
CSV_TIMESTAMP_FORMAT = "%Y-%m-%d"
CSV_FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# Console output
SEPARATOR_LENGTH = 60

# =============================================================================
# TRAINING SETTINGS (Janus / HyperLoRA)
# =============================================================================

# Default checkpoint paths
DEFAULT_JANUS_CHECKPOINT = "checkpoints/janus_v1/final"
DEFAULT_BASE_MODEL = "Qwen/Qwen2-7B-Instruct"

# HyperLoRA defaults
HYPERLORA_RANK = 16
HYPERLORA_ALPHA = 32.0
HYPERLORA_HIDDEN = 64
HYPERLORA_DROPOUT = 0.05

# Training defaults
TRAINING_BATCH_SIZE = 4
TRAINING_GRAD_ACCUM = 8
TRAINING_LR = 2e-4
TRAINING_MAX_STEPS = 20000

# =============================================================================
# DATASET GENERATION SETTINGS
# =============================================================================

DEFAULT_DATASET_OUTPUT = "datasets/price_domain.jsonl"
DEFAULT_PROCESSED_OUTPUT = "datasets/processed_tables"
K_HISTORY = 8  # Number of history items in training context

# =============================================================================
# DEBUG SETTINGS
# =============================================================================

DEBUG_MODE = False
VERBOSE_LOGGING = False
