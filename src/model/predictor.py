"""Inference and serving module for the trained model."""
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

from src.core.config import AppConfig, load_config
from src.model.dataset import LABEL_MAP, encode_text, load_vocab
from src.model.network import SentimentClassifierNet


class SentimentPredictor:
    """Production inference engine to score new input texts."""

    def __init__(self, config: AppConfig = None):
        self.config = config or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        save_dir = Path(self.config.training.model_save_dir)
        self.model_path = save_dir / self.config.training.model_save_name
        self.vocab_path = save_dir / self.config.training.vocab_save_name

        if not self.model_path.exists() or not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint or vocab not found in '{save_dir}'. "
                f"Please train the model first by running `python main.py train`."
            )

        # Load vocabulary
        self.vocab = load_vocab(str(self.vocab_path))

        # Reconstruct network
        self.model = SentimentClassifierNet(
            vocab_size=len(self.vocab),
            embedding_dim=self.config.model.embedding_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_classes=self.config.model.num_classes,
            dropout=0.0  # Turn off dropout during inference
        ).to(self.device)

        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment/intent for a single text string."""
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment/intent for a batch of strings."""
        if not texts:
            return []

        # Encode input texts into tensors
        encoded = [encode_text(t, self.vocab) for t in texts]
        input_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu()

        predictions = []
        for i, text in enumerate(texts):
            prob_list = probs[i].tolist()
            pred_class = int(torch.argmax(probs[i]).item())
            confidence = prob_list[pred_class]

            predictions.append({
                "text": text,
                "label": LABEL_MAP.get(pred_class, "Unknown"),
                "class_id": pred_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    LABEL_MAP[c]: round(prob_list[c], 4) for c in range(len(prob_list))
                },
            })

        return predictions
