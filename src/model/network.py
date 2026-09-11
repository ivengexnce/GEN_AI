"""PyTorch Neural Network Architecture for Text Classification."""
import torch
import torch.nn as nn
from src.model.dataset import PAD_IDX


class SentimentClassifierNet(nn.Module):
    """
    Feedforward Neural Network with Learnable Word Embeddings.

    Architecture:
    1. Embedding Layer: Maps discrete token IDs into dense vector representations.
    2. Global Average Pooling: Aggregates token vectors across the sequence dimension (ignoring padding).
    3. Hidden Dense Layer: Feature transformation with non-linear activation (ReLU).
    4. Dropout Layer: Regularization to prevent overfitting.
    5. Output Projection: Produces unnormalized class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            logits: FloatTensor of shape (batch_size, num_classes)
        """
        # Step 1: Lookup embeddings -> Shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # Step 2: Masked global average pooling (avoid averaging pad tokens)
        # Create mask where 1 = actual token, 0 = padding token
        mask = (input_ids != PAD_IDX).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        sum_embedded = (embedded * mask).sum(dim=1)          # (batch_size, embedding_dim)
        lengths = mask.sum(dim=1).clamp(min=1.0)             # (batch_size, 1)
        pooled = sum_embedded / lengths                      # (batch_size, embedding_dim)

        # Step 3 & 4: Hidden transformations + Dropout
        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)

        # Step 5: Class logits
        logits = self.fc2(hidden)
        return logits
