# LLM Chatbot

A PyTorch-based Large Language Model chatbot from scratch.

## Features

- Custom transformer-based language model
- Training pipeline with data loading
- Interactive chatbot inference
- Tokenization and vocabulary management
- Checkpoint saving and loading

## Project Structure

```
llm-chatbot/
├── data/                 # Training data
│   └── sample_data.txt
├── models/               # Model architectures
│   ├── __init__.py
│   ├── transformer.py
│   └── tokenizer.py
├── src/                  # Core training and inference
│   ├── __init__.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── checkpoints/          # Saved model weights
├── requirements.txt      # Python dependencies
├── config.yaml           # Configuration file
└── main.py              # Entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Rmb45/llm-chatbot.git
cd llm-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python main.py --mode train --data data/sample_data.txt --epochs 10
```

### Chatbot Inference

```bash
python main.py --mode chat --checkpoint checkpoints/latest.pt
```

### Generate Text

```bash
python main.py --mode generate --prompt "Hello, how are you?"
```

## Configuration

Edit `config.yaml` to adjust:
- Model dimensions (hidden size, num layers)
- Training parameters (learning rate, batch size)
- Data settings (sequence length, vocab size)

## Model Architecture

The model uses a transformer-based architecture:
- **Multi-head Self-Attention**: Allows the model to attend to different positions
- **Feed-Forward Networks**: Provides non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Positional Encoding**: Encodes token positions in the sequence
- **Causal Masking**: Prevents attention to future tokens (for language modeling)

## Components

### Tokenizer (`models/tokenizer.py`)
- Word-level tokenization
- Vocabulary management
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

### Transformer Model (`models/transformer.py`)
- Positional encoding
- Multi-head attention
- Feed-forward layers
- Transformer blocks with residual connections

### Training (`src/train.py`)
- Custom text dataset loader
- Training loop with validation
- Checkpoint saving

### Inference (`src/inference.py`)
- Text generation with temperature sampling
- Interactive chat interface

## Usage Examples

### Train a new model
```bash
python main.py --mode train --epochs 20 --data data/sample_data.txt
```

### Chat with trained model
```bash
python main.py --mode chat
```

### Generate text from a prompt
```bash
python main.py --mode generate --prompt "Once upon a time"
```

## Hyperparameters

Key hyperparameters in `config.yaml`:
- `vocab_size`: 10000
- `max_seq_length`: 512
- `d_model`: 512 (embedding dimension)
- `num_layers`: 6 (number of transformer blocks)
- `num_heads`: 8 (attention heads)
- `d_ff`: 2048 (feed-forward dimension)
- `dropout`: 0.1
- `learning_rate`: 0.0001
- `batch_size`: 32

## Tips for Better Results

1. **Data Quality**: Provide high-quality training data with diverse content
2. **Vocabulary Size**: Adjust `vocab_size` based on your data (10000 is a good start)
3. **Sequence Length**: Longer sequences capture more context but require more memory
4. **Training Time**: More epochs = better results, but watch for overfitting
5. **Temperature**: Lower values (0.5-0.7) for coherent text, higher (0.8-1.0) for creativity

## License

MIT
