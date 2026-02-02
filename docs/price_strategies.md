# Price Domain Strategy Registry

This document lists the deterministic strategies available for offline RL dataset generation and evaluation.

These strategies are implemented in `src/agents/price_strategies.py` and registered in `STRATEGY_REGISTRY`.

## Overview

The dataset generation mode can use any combination of these strategies. This ensures datasets contain diverse negotiation behaviors, from cooperative to competitive.

## Strategies

### Boulware Agents (Spectrum)

Agents that follow a time-dependent concession curve:

$$P(t) = Start + (Reservation - Start) \times (t/T)^\beta$$

| Strategy | Beta | Description |
|----------|------|-------------|
| `boulware_very_conceding` | 0.2 | Rapidly concedes early |
| `boulware_conceding` | 0.5 | Moderately concedes early |
| `boulware_linear` | 1.0 | Linear concession over time |
| `boulware_firm` | 2.0 | Holds value until late |
| `boulware_hard` | 4.0 | Very slow concession |

### Fixed Strategies

| Strategy | Margin | Description |
|----------|--------|-------------|
| `price_fixed_strict` | 5.0 | Offers exactly Reservation ± 5 |
| `price_fixed_loose` | 25.0 | Offers exactly Reservation ± 25 |

### Adaptive Strategies

| Strategy | Description |
|----------|-------------|
| `tit_for_tat` | Mirrors opponent's last concession magnitude |
| `split_difference` | Proposes midpoint between offers |
| `time_dependent` | Relaxes acceptance threshold over time |
| `hardliner` | Static hardline until final round |
| `linear` | Standard linear concession |

### MiCRO Variants

MiCRO (Minimal Concession with Risk for Opponent) strategies:

| Strategy | Initial Opening | Description |
|----------|-----------------|-------------|
| `micro_low` | Res ± 100 | Conservative opening |
| `micro_mid` | Res ± 200 | Moderate opening |
| `micro_high` | Res ± 300 | Aggressive opening |

### Oracle Strategy

| Strategy | Description |
|----------|-------------|
| `random_zopa` | Random price within true ZOPA (requires oracle access) |

## Usage

### Dataset Generation

```bash
python negotiation.py --buyer_strategy boulware_conceding --seller_strategy linear \
    --num_runs 1000 --dataset_out datasets/price_domain.jsonl
```

### Running Experiments

```bash
python negotiation.py --buyer_strategy tit_for_tat --seller_strategy hardliner \
    --num_runs 100 --output logs/experiment.csv
```

### Janus vs Deterministic

```bash
python run_janus_vs_deterministic.py --num_episodes 10 --janus_buyer_rho 0.3
```

## Strategy Implementation Details

All strategies implement the following interface:

```python
def propose_action(self, state: PriceState) -> PriceAction:
    """
    Given current negotiation state, return action.
    
    Args:
        state: Current PriceState with turn, history, reservation, etc.
        
    Returns:
        PriceAction with type="OFFER"|"ACCEPT" and optional price
    """
```

### Key State Fields

- `timestep`: Current turn number (1-indexed)
- `max_turns`: Maximum turns in episode
- `role`: "buyer" or "seller"
- `last_offer_price`: Opponent's most recent offer (or None)
- `offer_history`: List of (role, price) tuples
- `true_reservation_price`: Agent's private reservation value

### Accept Logic

Most strategies check for acceptance before making an offer:

```python
# Buyer accepts if offer <= reservation
if role == "buyer" and last_offer <= reservation:
    return PriceAction(type="ACCEPT")

# Seller accepts if offer >= reservation  
if role == "seller" and last_offer >= reservation:
    return PriceAction(type="ACCEPT")
```

## Adding New Strategies

1. Define strategy function in `src/agents/price_strategies.py`:

```python
def my_strategy(state: PriceState, params: Dict) -> Tuple[str, float]:
    """Return (action_type, price)."""
    # Your logic here
    return "OFFER", calculated_price
```

2. Register in `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY["my_strategy"] = {
    "function": my_strategy,
    "params": {"param1": value1}
}
```

3. Use in experiments:

```bash
python negotiation.py --buyer_strategy my_strategy --seller_strategy linear
```
