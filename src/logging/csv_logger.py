"""
CSV Logger for negotiation experiments.

Logs experiment results to CSV files for analysis.
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List


class CSVLogger:
    """
    Logs negotiation experiment results to a CSV file.
    """
    
    # Default fieldnames for price negotiation experiments
    DEFAULT_FIELDNAMES = [
        'run_id',
        'timestamp',
        'buyer_strategy',
        'seller_strategy',
        'buyer_max',
        'seller_min',
        'zopa_width',
        'max_turns',
        'num_turns',
        'agreement',
        'final_price',
        'buyer_utility',
        'seller_utility',
        'rho',
        'buyer_type',
        'seller_type',
        'notes'
    ]
    
    def __init__(
        self, 
        filepath: str,
        fieldnames: Optional[List[str]] = None,
        append: bool = True
    ):
        """
        Initialize the CSV logger.
        
        Args:
            filepath: Path to the CSV file
            fieldnames: List of column names. If None, uses DEFAULT_FIELDNAMES
            append: If True, append to existing file. If False, overwrite.
        """
        self.filepath = filepath
        self.fieldnames = fieldnames or self.DEFAULT_FIELDNAMES
        
        # Ensure directory exists
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        # Check if file exists
        file_exists = os.path.exists(filepath)
        
        # Open file
        mode = 'a' if append else 'w'
        self._file = open(filepath, mode, newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction='ignore')
        
        # Write header if new file or overwriting
        if not file_exists or not append:
            self._writer.writeheader()
            self._file.flush()
            
    def log(self, **kwargs) -> None:
        """
        Log a single row to the CSV file.
        
        Args:
            **kwargs: Column name -> value pairs
        """
        # Add timestamp if not provided
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now().isoformat()
            
        self._writer.writerow(kwargs)
        self._file.flush()
        
    def log_negotiation_result(
        self,
        run_id: str,
        buyer_strategy: str,
        seller_strategy: str,
        buyer_max: float,
        seller_min: float,
        max_turns: int,
        num_turns: int,
        agreement: bool,
        final_price: Optional[float] = None,
        buyer_type: str = "deterministic",
        seller_type: str = "deterministic",
        notes: str = ""
    ) -> None:
        """
        Log a negotiation result with computed utilities.
        
        Args:
            run_id: Unique identifier for this run
            buyer_strategy: Name of buyer's strategy
            seller_strategy: Name of seller's strategy
            buyer_max: Buyer's maximum acceptable price
            seller_min: Seller's minimum acceptable price
            max_turns: Maximum number of turns allowed
            num_turns: Actual number of turns taken
            agreement: Whether an agreement was reached
            final_price: The agreed price (if agreement)
            buyer_type: Type of buyer agent (deterministic, llm, janus)
            seller_type: Type of seller agent (deterministic, llm, janus)
            notes: Additional notes
        """
        zopa_width = buyer_max - seller_min
        
        buyer_utility = None
        seller_utility = None
        rho = None
        
        if agreement and final_price is not None:
            # Buyer utility: how much below their max they paid
            buyer_utility = buyer_max - final_price
            # Seller utility: how much above their min they received  
            seller_utility = final_price - seller_min
            # Rho: normalized position within ZOPA
            if zopa_width > 0:
                rho = (final_price - seller_min) / zopa_width
                
        self.log(
            run_id=run_id,
            buyer_strategy=buyer_strategy,
            seller_strategy=seller_strategy,
            buyer_max=buyer_max,
            seller_min=seller_min,
            zopa_width=zopa_width,
            max_turns=max_turns,
            num_turns=num_turns,
            agreement=agreement,
            final_price=final_price,
            buyer_utility=buyer_utility,
            seller_utility=seller_utility,
            rho=rho,
            buyer_type=buyer_type,
            seller_type=seller_type,
            notes=notes
        )
        
    def close(self) -> None:
        """Close the CSV file."""
        if self._file:
            self._file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
