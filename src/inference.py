import torch
from typing import List

def generate_text(model: torch.nn.Module, tokenizer, prompt: str, max_length: int = 100,
                 temperature: float = 0.7, device: torch.device = torch.device('cpu')) -> str:
    """Generate text given a prompt."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-512:]  # Limit to max sequence length
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(input_tensor)
            
            # Get last token logits
            next_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
            
            # Stop if EOS token
            if next_token == tokenizer.word2idx.get('<EOS>', 3):
                break
    
    # Decode and return
    generated_text = tokenizer.decode(tokens)
    return generated_text

def chat(model: torch.nn.Module, tokenizer, device: torch.device = torch.device('cpu')):
    """Interactive chatbot."""
    print("\n=== LLM Chatbot ===")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Generate response
        response = generate_text(model, tokenizer, user_input, max_length=50, device=device)
        print(f"Bot: {response}\n")
