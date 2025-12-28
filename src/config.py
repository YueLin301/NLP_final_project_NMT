import torch

class Config:
    # Data paths
    TRAIN_FILE = 'data/train_100k.jsonl' # Default to larger set, can switch to 10k
    VALID_FILE = 'data/valid.jsonl'
    TEST_FILE = 'data/test.jsonl'
    
    # Model checkpoints
    MODEL_SAVE_DIR = 'checkpoints'
    
    # Tokenizer
    SRC_LANG = 'zh'
    TGT_LANG = 'en'
    
    # Vocab
    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 30000
    
    # Model Hyperparameters
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.5
    
    # Debug/Test mode
    DEBUG = False
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    def __init__(self):
        pass
