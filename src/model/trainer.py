"""Training pipeline for the Neural Network model."""
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.core.config import AppConfig
from src.model.dataset import (
    TextSentimentDataset,
    build_vocab_from_texts,
    get_default_training_data,
    save_vocab,
)
from src.model.network import SentimentClassifierNet


class ModelTrainer:
    """Orchestrates model training, evaluation, checkpointing, and artifact export."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(
        self, texts: List[str], labels: List[int]
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Tokenize, build vocabulary, and create train/validation data loaders."""
        vocab = build_vocab_from_texts(texts, max_vocab_size=self.config.model.vocab_size)

        dataset = TextSentimentDataset(texts, labels, vocab)
        total_size = len(dataset)
        train_size = int(self.config.data.train_split * total_size)
        val_size = total_size - train_size

        # Deterministic split for reproducible training
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, vocab

    def train(
        self, custom_texts: List[str] = None, custom_labels: List[int] = None
    ) -> Dict[str, float]:
        """Execute the end-to-end training cycle."""
        if custom_texts is None or custom_labels is None:
            texts, labels = get_default_training_data()
        else:
            texts, labels = custom_texts, custom_labels

        train_loader, val_loader, vocab = self.prepare_data(texts, labels)

        # Initialize network
        model = SentimentClassifierNet(
            vocab_size=len(vocab),
            embedding_dim=self.config.model.embedding_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_classes=self.config.model.num_classes,
            dropout=self.config.model.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.training.learning_rate
        )

        print(f"[*] Training on device: {self.device}")
        print(f"[*] Vocabulary size: {len(vocab)} tokens")
        print(f"[*] Training samples: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}")

        epochs = self.config.training.epochs
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # --- Training phase ---
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for input_ids, targets in train_loader:
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += len(targets)

            train_loss = total_loss / total
            train_acc = correct / total

            # --- Validation phase ---
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)

            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"| Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% "
                    f"| Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%"
                )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(model, vocab)

        print(f"[+] Model training completed. Best Val Acc: {best_val_acc*100:.1f}%")
        return {"train_acc": train_acc, "val_acc": val_acc, "best_val_acc": best_val_acc}

    def evaluate(
        self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model performance without computing gradients."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, targets in data_loader:
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)
                logits = model(input_ids)
                loss = criterion(logits, targets)

                total_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def _save_checkpoint(self, model: nn.Module, vocab: Dict[str, int]) -> None:
        """Persist model state and vocabulary for serving."""
        save_dir = Path(self.config.training.model_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / self.config.training.model_save_name
        vocab_path = save_dir / self.config.training.vocab_save_name

        torch.save(model.state_dict(), model_path)
        save_vocab(vocab, str(vocab_path))
