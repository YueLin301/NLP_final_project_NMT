import argparse
import subprocess
from src.config import Config

def main():
    print("Running in DEBUG mode (very fast training)...")
    
    # 1. Modify Config to DEBUG mode temporary or just pass a flag if scripts supported it.
    # Since our scripts read Config(), we can't easily injection without modifying file.
    # BUT, we can add a --debug flag to our training scripts!
    
    # Let's run a single experiment with few epochs and debug flag
    
    # We need to update train_rnn.py and train_transformer.py to accept --debug
    # I will assume I'll update them next.
    
    cmd = "python train_rnn.py --attention dot --n_epochs 1 --save_name rnn_debug.pt --debug"
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n✅ RNN Debug run successful!")
    except subprocess.CalledProcessError:
        print("\n❌ RNN Debug run failed!")
        
    cmd = "python train_transformer.py --norm_type layernorm --n_epochs 1 --save_name trans_debug.pt --debug"
    print(f"\nRunning: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n✅ Transformer Debug run successful!")
    except subprocess.CalledProcessError:
        print("\n❌ Transformer Debug run failed!")

if __name__ == '__main__':
    main()

