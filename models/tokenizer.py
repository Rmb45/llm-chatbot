import json
from collections import Counter
from typing import List, Dict

class SimpleTokenizer:
    """Simple word-level tokenizer with vocabulary management."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_count = Counter()
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens."""
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.word2idx['<BOS>'] = 2  # Beginning of sequence
        self.word2idx['<EOS>'] = 3  # End of sequence
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        for text in texts:
            tokens = text.lower().split()
            self.vocab_count.update(tokens)
        
        # Keep top vocab_size words
        idx = len(self.word2idx)
        for word, _ in self.vocab_count.most_common(self.vocab_size - len(self.word2idx)):
            if word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = text.lower().split()
        ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.idx2word.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.vocab_size = data['vocab_size']
            self.idx2word = {v: k for k, v in self.word2idx.items()}
