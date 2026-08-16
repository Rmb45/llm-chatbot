import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing import Tuple
import numpy as np

class TextDataset(Dataset):
    """Custom dataset for text data."""
    
    def __init__(self, texts: list, tokenizer, max_seq_length: int, pad_idx: int = 0):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_length - 1:
            tokens = tokens[:self.max_seq_length - 1]
        
        # Prepare input and target
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Pad
        padding_length = self.max_seq_length - 1 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_idx] * padding_length
            target_ids = target_ids + [self.pad_idx] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def load_data(filepath: str, batch_size: int = 32, max_seq_length: int = 512,
             train_split: float = 0.8, tokenizer=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare data."""
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    sentences = text.split('.')
    texts = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    # Build tokenizer vocabulary
    if tokenizer is not None:
        tokenizer.build_vocab(texts)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_seq_length)
    
    # Split into train, val, test
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
               criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device,
               epochs: int = 10, checkpoint_dir: str = 'checkpoints'):
    """Train model for multiple epochs."""
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            from .utils import save_checkpoint
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
