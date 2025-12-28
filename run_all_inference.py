import os
import subprocess
import glob

def main():
    print("="*50)
    print("🚀 Batch Inference on All Trained Models")
    print("="*50 + "\n")
    
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        print("Checkpoints directory not found!")
        return

    # Find all model files
    model_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
    # Filter out vocab files and debug models
    model_files = [f for f in model_files if 'vocab' not in f and 'debug' not in f]
    model_files.sort()
    
    if not model_files:
        print("No models found in checkpoints/")
        return

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        
        # Determine model type based on filename
        if 'rnn' in model_name:
            model_type = 'rnn'
        elif 'trans' in model_name:
            model_type = 'transformer'
        else:
            print(f"Skipping unknown model type: {model_name}")
            continue
            
        print(f"\n🔍 Testing Model: {model_name} ({model_type})")
        print("-" * 40)
        
        cmd = f"python inference.py --model_type {model_type} --model_path {model_path}"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Failed to run inference on {model_name}")
        
        print("\n")

if __name__ == '__main__':
    main()

