import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def plot_comparison(json_files, output_file, metric='valid_loss', title='Comparison', labels=None):
    plt.figure(figsize=(10, 6))
    
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        y_values = data.get(metric, [])
        x_values = range(1, len(y_values) + 1)
        
        label = labels[i] if labels else os.path.basename(json_file).replace('.json', '')
        plt.plot(x_values, y_values, label=label, marker='o')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()

def main():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("Results directory not found.")
        return

    # 1. Compare RNN Attention Mechanisms
    rnn_attentions = ['rnn_dot.json', 'rnn_general.json', 'rnn_concat.json']
    rnn_files = [os.path.join(results_dir, f) for f in rnn_attentions if os.path.exists(os.path.join(results_dir, f))]
    
    if rnn_files:
        plot_comparison(rnn_files, os.path.join(results_dir, 'comparison_rnn_attention_loss.png'), 
                        metric='valid_loss', title='RNN Attention: Validation Loss')
        plot_comparison(rnn_files, os.path.join(results_dir, 'comparison_rnn_attention_bleu.png'), 
                        metric='valid_bleu', title='RNN Attention: Validation BLEU')

    # 2. Compare Transformer Normalization
    trans_norms = ['trans_layernorm.json', 'trans_rmsnorm.json']
    trans_files = [os.path.join(results_dir, f) for f in trans_norms if os.path.exists(os.path.join(results_dir, f))]
    
    if trans_files:
        plot_comparison(trans_files, os.path.join(results_dir, 'comparison_transformer_norm_loss.png'), 
                        metric='valid_loss', title='Transformer Norm: Validation Loss')
        plot_comparison(trans_files, os.path.join(results_dir, 'comparison_transformer_norm_bleu.png'), 
                        metric='valid_bleu', title='Transformer Norm: Validation BLEU')

    # 3. Compare Best RNN vs Best Transformer (Assuming Dot and LayerNorm as baselines or check which exists)
    best_rnn = 'rnn_concat.json' if os.path.exists(os.path.join(results_dir, 'rnn_concat.json')) else 'rnn_dot.json'
    best_trans = 'trans_layernorm.json'
    
    compare_files = [os.path.join(results_dir, f) for f in [best_rnn, best_trans] if os.path.exists(os.path.join(results_dir, f))]
    
    if len(compare_files) >= 2:
        plot_comparison(compare_files, os.path.join(results_dir, 'comparison_rnn_vs_transformer_bleu.png'),
                        metric='valid_bleu', title='RNN vs Transformer: BLEU')

if __name__ == '__main__':
    main()

