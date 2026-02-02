
import sys
import os
import subprocess

def main():
    """
    Wrapper script to run HyperLoRA training from the root of the SPP domain.
    Ensures that the 'src' package is discoverable.
    """
    # Add current directory to PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Check if arguments are provided
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python train.py [arguments]")
        print("Example: python train.py --decision_steps_path datasets/processed_tables/decision_steps.parquet --output_dir checkpoints/new_run")
        print("\nPass --help to see all available options from the underlying training script.")
        
        # If no args, we can default to help to be helpful
        args = ["--help"]
        
    # Construct the command to run the actual training script
    # We use running as a module 'src.training.train_janus_hyperlora' to handle imports correctly
    cmd = [sys.executable, "-m", "src.training.train_janus_hyperlora"] + args
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.call(cmd)

if __name__ == "__main__":
    main()
