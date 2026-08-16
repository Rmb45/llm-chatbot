import argparse
import torch
import torch.nn as nn
from models.transformer import TransformerLM
from models.tokenizer import SimpleTokenizer
from src.utils import set_seed, load_config, get_device, load_checkpoint
from src.train import load_data, train_model
from src.inference import generate_text, chat

def main():
    parser = argparse.ArgumentParser(description='LLM Chatbot')
    parser.add_argument('--mode', choices=['train', 'chat', 'generate'], default='chat',
                       help='Mode: train, chat, or generate')
    parser.add_argument('--data', type=str, default='data/sample_data.txt',
                       help='Path to training data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pt',
                       help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--prompt', type=str, default='Hello',
                       help='Prompt for text generation')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config['model']['vocab_size'])
    
    # Initialize model
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        max_seq_length=config['model']['max_seq_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    ).to(device)
    
    if args.mode == 'train':
        print("\n=== Starting Training ===")
        
        # Load data
        train_loader, val_loader, test_loader = load_data(
            args.data,
            batch_size=config['training']['batch_size'],
            max_seq_length=config['model']['max_seq_length'],
            train_split=config['data']['train_split'],
            tokenizer=tokenizer
        )
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Train
        train_model(
            model, train_loader, val_loader, optimizer, criterion, device,
            epochs=args.epochs, checkpoint_dir='checkpoints'
        )
        
        print("\nTraining completed!")
    
    elif args.mode == 'chat':
        print("\n=== Loading Chatbot ===")
        
        # Load checkpoint
        try:
            load_checkpoint(args.checkpoint, model)
            # Load tokenizer - create from existing vocab
            tokenizer.vocab_size = config['model']['vocab_size']
            print("Model and tokenizer loaded.")
        except FileNotFoundError:
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Please train the model first using: python main.py --mode train")
            return
        
        # Start chat
        chat(model, tokenizer, device=device)
    
    elif args.mode == 'generate':
        print("\n=== Generating Text ===")
        
        # Load checkpoint
        try:
            load_checkpoint(args.checkpoint, model)
            print("Model loaded.")
        except FileNotFoundError:
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        
        # Generate text
        generated = generate_text(model, tokenizer, args.prompt, max_length=100, device=device)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")

if __name__ == '__main__':
    main()
