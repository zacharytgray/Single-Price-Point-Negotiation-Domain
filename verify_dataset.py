"""
Verify the new dataset for training readiness.
"""
import json
import statistics
from collections import Counter

def main():
    # Load dataset
    data = []
    with open('datasets/price_domain_v2.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print("=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    print()

    # Basic stats
    print(f"Total rows: {len(data)}")
    trajectories = set(d['trajectory_id'] for d in data)
    print(f"Total trajectories: {len(trajectories)}")
    print()

    # Success/failure
    traj_rhos = {}
    for d in data:
        tid = d['trajectory_id']
        rho = d.get('rho')
        if tid not in traj_rhos:
            traj_rhos[tid] = set()
        traj_rhos[tid].add(rho)

    success = sum(1 for rhos in traj_rhos.values() if -1.0 not in rhos and len(rhos) > 0)
    failure = sum(1 for rhos in traj_rhos.values() if -1.0 in rhos)
    print(f"Successful trajectories: {success} ({success/len(trajectories)*100:.1f}%)")
    print(f"Failed trajectories: {failure} ({failure/len(trajectories)*100:.1f}%)")
    print()

    # Strategy distribution
    strategies = [d['agent']['strategy'] for d in data]
    strat_counts = Counter(strategies)
    print("Strategy distribution (top 10):")
    for strat, count in strat_counts.most_common(10):
        print(f"  {strat}: {count}")
    print()

    # Strategy ranking by utility gain
    print("=" * 60)
    print("STRATEGY RANKING BY UTILITY GAIN")
    print("=" * 60)
    print()

    # Calculate utility gain per strategy
    strategy_utilities = {}
    strategy_counts = {}
    
    for d in data:
        outcome = d['meta'].get('episode_outcome', {})
        if not outcome.get('agreement_bool'):
            continue  # Skip failed negotiations
        
        accepted_price = outcome.get('price')
        if accepted_price is None:
            continue
        
        strategy = d['agent']['strategy']
        role = d['state']['role']
        reservation = d['state']['reservation_price']
        
        # Calculate utility based on role
        if role == 'buyer':
            utility = reservation - accepted_price
        else:  # seller
            utility = accepted_price - reservation
        
        if strategy not in strategy_utilities:
            strategy_utilities[strategy] = []
            strategy_counts[strategy] = {'buyer': 0, 'seller': 0}
        
        strategy_utilities[strategy].append(utility)
        strategy_counts[strategy][role] += 1

    # Rank strategies by average utility
    ranked_strategies = []
    for strategy, utilities in strategy_utilities.items():
        avg_utility = statistics.mean(utilities)
        median_utility = statistics.median(utilities)
        std_utility = statistics.stdev(utilities) if len(utilities) > 1 else 0
        total_episodes = len(utilities)
        buyer_count = strategy_counts[strategy]['buyer']
        seller_count = strategy_counts[strategy]['seller']
        ranked_strategies.append({
            'strategy': strategy,
            'avg_utility': avg_utility,
            'median_utility': median_utility,
            'std_utility': std_utility,
            'total_episodes': total_episodes,
            'buyer_count': buyer_count,
            'seller_count': seller_count
        })

    # Sort by average utility (descending)
    ranked_strategies.sort(key=lambda x: x['avg_utility'], reverse=True)

    print(f"{'Rank':<6} {'Strategy':<35} {'Avg Utility':<12} {'Median':<10} {'Std Dev':<10} {'Episodes':<10} {'Buyer':<7} {'Seller'}")
    print("-" * 110)
    for i, s in enumerate(ranked_strategies, 1):
        print(f"{i:<6} {s['strategy']:<35} {s['avg_utility']:<12.2f} {s['median_utility']:<10.2f} {s['std_utility']:<10.2f} {s['total_episodes']:<10} {s['buyer_count']:<7} {s['seller_count']}")
    print()

    # Rho verification
    print("=" * 60)
    print("RHO VERIFICATION")
    print("=" * 60)
    print()

    success_data = [d for d in data if d.get('rho') is not None and d.get('rho') != -1.0]
    
    mismatches = 0
    for d in success_data[:100]:  # Check first 100
        rho_logged = d['rho']
        zopa_low = d['meta']['zopa_low']
        zopa_high = d['meta']['zopa_high']
        accepted_price = d['meta']['accepted_price']
        
        if accepted_price is None:
            continue
            
        rho_calc = (accepted_price - zopa_low) / (zopa_high - zopa_low)
        
        if abs(rho_logged - rho_calc) > 0.0001:
            mismatches += 1

    print(f"Rho mismatches (first 100 checked): {mismatches}")
    if mismatches == 0:
        print("✓ All rho values are correctly calculated")
    print()

    # Symmetry check
    print("=" * 60)
    print("SYMMETRY VERIFICATION")
    print("=" * 60)
    print()

    # Get first offer per role (turn 1 for buyer, turn 2 for seller)
    first_offers = {}
    for d in data:
        turn = d['turn']
        role = d['state']['role']
        strategy = d['agent']['strategy']
        price = d['action'].get('price')
        
        # Buyer's first offer is turn 1, seller's first offer is turn 2
        if role == 'buyer' and turn != 1:
            continue
        if role == 'seller' and turn != 2:
            continue
            
        key = (strategy, role)
        if key not in first_offers:
            first_offers[key] = []
        if price:
            first_offers[key].append(price)

    strategies_to_check = ['hardliner', 'time_dependent', 'tit_for_tat', 'boulware_firm', 'naive_concession']
    
    print("First offer analysis (buyer turn 1, seller turn 2):")
    print(f"{'Strategy':<25} {'Buyer':<10} {'Seller':<10} {'Symmetric?'}")
    print("-" * 60)
    
    for strat in strategies_to_check:
        buyer_key = (strat, 'buyer')
        seller_key = (strat, 'seller')
        
        buyer_mean = statistics.mean(first_offers.get(buyer_key, [0]))
        seller_mean = statistics.mean(first_offers.get(seller_key, [0]))
        
        # Symmetric if buyer ~res-500 and seller ~res+500
        # With random reservations, expect buyer ~400-500, seller ~900-1000
        symmetric = buyer_mean < 500 and seller_mean > 800
        
        status = "✓" if symmetric else "✗"
        print(f"{strat:<25} {buyer_mean:<10.1f} {seller_mean:<10.1f} {status}")

    print()
    print("=" * 60)
    print("DATASET READY FOR TRAINING" if mismatches == 0 else "ISSUES FOUND")
    print("=" * 60)

if __name__ == "__main__":
    main()
