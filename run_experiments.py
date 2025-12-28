import os
import subprocess
import argparse

def run_experiment(command, description):
    print(f"\n{'='*20}\nRunning: {description}\nCommand: {command}\n{'='*20}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run NMT Experiments')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per experiment (default: 5 for speed)')
    args = parser.parse_args()
    
    epochs = args.epochs
    
    # 1. RNN Experiments
    # 1.1 Attention Mechanisms
    run_experiment(f"python train_rnn.py --attention dot --n_epochs {epochs} --save_name rnn_dot.pt", "RNN with Dot Attention")
    run_experiment(f"python train_rnn.py --attention general --n_epochs {epochs} --save_name rnn_general.pt", "RNN with General Attention")
    run_experiment(f"python train_rnn.py --attention concat --n_epochs {epochs} --save_name rnn_concat.pt", "RNN with Additive (Concat) Attention")
    
    # 1.2 Training Strategies (Teacher Forcing vs Free Running - approximated by low TF ratio)
    run_experiment(f"python train_rnn.py --teacher_forcing 0.1 --n_epochs {epochs} --save_name rnn_free.pt", "RNN with Low Teacher Forcing (Free Running-ish)")
    # High TF is default (0.5 or higher in code)
    
    # 2. Transformer Experiments
    # 2.1 Normalization
    run_experiment(f"python train_transformer.py --norm_type layernorm --n_epochs {epochs} --save_name trans_layernorm.pt", "Transformer with LayerNorm")
    run_experiment(f"python train_transformer.py --norm_type rmsnorm --n_epochs {epochs} --save_name trans_rmsnorm.pt", "Transformer with RMSNorm")
    
    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()

