from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BigramConfig:
    context_length: int = 1
    path_to_data: Path = Path("data")
    to_log: bool = False
    log_interval: int = 100
    save_path: Path = Path("models/bigram/")
    batch_size: int = 32
    scheduler: bool = False
    to_clip_grad: bool = False
    gradient_clip: float = 1.0
    vocab_size: int = 50257
    embed_dim: int = 32
    dropout: float = 0.1
    save_iterations: int = 10000
    max_iter: int = 500000
    learning_rate: float = 3e-2


@dataclass
class MiniGPTConfig:
    path_to_data: Path = Path("data")
    # batch_size: int = 10
    batch_size: int = 10
    # num_layers: int = 2  # Num of Transformer layers
    num_layers: int = 1
    vocab_size: int = 50257  # Tiktoken for GPT2 size
    # vocab_size: int = 10000
    embed_dim: int = (
        64  # Dimensionality of the the token embeddings (throught the transformer)
    )
    feedforward_size: Optional[int] = (
        None  # hidden layer in the feedforward network, None sets it to 4*embed_dim
    )
    context_length: int = 10
    # context_length: int = 512  # Max number of tokens in a sequence
    num_heads: int = 4  # Number of heads in the multihead attention
    weight_tie: bool = (
        True  # Whether to tie the weights of the embedding and the output layer
    )
    feedforward_dropout: float = 0.1  # Dropout in the feedforward layer
    attention_dropout: float = 0.1  # Dropout in the attention layer
    out_dropout: float = 0.1  # Dropout in the output layer
    embed_dropout: float = 0.1  # Dropout in the embedding layer
    learning_rate: float = 3e-4  # Learning rate for the optimizer
    log_interval: int = 10
    save_path: Path = Path("models/minigpt/")
    # save_iterations: int = 10000
    save_iterations: int = 100
    to_log: bool = False
    # max_iter: int = 5000
    max_iter: int = 1
    to_clip_grad: bool = False
    gradient_clip: float = 1.0
    scheduler: bool = False
