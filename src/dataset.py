import json
import jieba
import torch
from torch.utils.data import Dataset
from collections import Counter
import re

# Simple English tokenizer using regex if nltk is not preferred or for simplicity
def tokenize_en(text):
    # Basic tokenization: split by non-alphanumeric, keep punctuation
    return [tok for tok in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]

def tokenize_zh(text):
    return list(jieba.cut(text))

class Vocabulary:
    def __init__(self, freq_threshold=2, max_size=30000):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in tokenizer(sentence):
                frequencies[word] += 1
        
        # Sort by frequency and truncate
        common_words = frequencies.most_common(self.max_size - 4)
        
        for word, count in common_words:
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text, tokenizer):
        tokenized_text = tokenizer(text)
        
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokenized_text
        ]

class NMTDataset(Dataset):
    def __init__(self, file_path, src_vocab=None, tgt_vocab=None, src_tokenizer=tokenize_zh, tgt_tokenizer=tokenize_en, build_vocab=False, max_len=100):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        if build_vocab:
            self.src_vocab = Vocabulary()
            self.tgt_vocab = Vocabulary()
            
            src_texts = [item['zh'] for item in self.data]
            tgt_texts = [item['en'] for item in self.data]
            
            print("Building Source Vocabulary...")
            self.src_vocab.build_vocabulary(src_texts, self.src_tokenizer)
            print("Building Target Vocabulary...")
            self.tgt_vocab.build_vocabulary(tgt_texts, self.tgt_tokenizer)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        src_text = item['zh']
        tgt_text = item['en']
        
        src_numericalized = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.numericalize(src_text, self.src_tokenizer) + [self.src_vocab.stoi["<eos>"]]
        tgt_numericalized = [self.tgt_vocab.stoi["<sos>"]] + self.tgt_vocab.numericalize(tgt_text, self.tgt_tokenizer) + [self.tgt_vocab.stoi["<eos>"]]
        
        # Truncate if necessary (though batching usually handles this)
        if len(src_numericalized) > self.max_len:
            src_numericalized = src_numericalized[:self.max_len-1] + [self.src_vocab.stoi["<eos>"]]
        if len(tgt_numericalized) > self.max_len:
            tgt_numericalized = tgt_numericalized[:self.max_len-1] + [self.tgt_vocab.stoi["<eos>"]]

        return torch.tensor(src_numericalized), torch.tensor(tgt_numericalized)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch]
        tgt = [item[1] for item in batch]
        
        src_pad = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=self.pad_idx)
        
        return src_pad, tgt_pad

