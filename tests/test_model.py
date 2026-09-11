"""Unit tests for the Neural Network model, dataset, and predictor."""
import unittest
import torch
from src.core.config import AppConfig
from src.model.dataset import (
    tokenize,
    build_vocab_from_texts,
    encode_text,
    TextSentimentDataset,
    get_default_training_data,
)
from src.model.network import SentimentClassifierNet
from src.model.trainer import ModelTrainer
from src.model.predictor import SentimentPredictor


class TestModelPipeline(unittest.TestCase):

    def test_tokenization_and_vocab(self):
        texts = ["Hello world!", "AI agent development is awesome."]
        tokens = tokenize(texts[0])
        self.assertEqual(tokens, ["hello", "world"])

        vocab = build_vocab_from_texts(texts)
        self.assertIn("<pad>", vocab)
        self.assertIn("<unk>", vocab)
        self.assertIn("hello", vocab)
        self.assertIn("awesome", vocab)

    def test_encode_text(self):
        vocab = {"<pad>": 0, "<unk>": 1, "great": 2, "model": 3}
        encoded = encode_text("great model test", vocab, max_len=5)
        self.assertEqual(len(encoded), 5)
        self.assertEqual(encoded[0], 2)  # "great"
        self.assertEqual(encoded[1], 3)  # "model"
        self.assertEqual(encoded[2], 1)  # "test" -> <unk>
        self.assertEqual(encoded[3], 0)  # pad
        self.assertEqual(encoded[4], 0)  # pad

    def test_network_forward_pass(self):
        batch_size = 4
        seq_len = 10
        vocab_size = 50
        num_classes = 3

        model = SentimentClassifierNet(
            vocab_size=vocab_size,
            embedding_dim=16,
            hidden_dim=8,
            num_classes=num_classes,
            dropout=0.0
        )
        fake_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(fake_input)

        self.assertEqual(logits.shape, (batch_size, num_classes))

    def test_trainer_and_predictor(self):
        # Quick mini training run
        config = AppConfig()
        config.training.epochs = 2
        config.training.batch_size = 4

        trainer = ModelTrainer(config)
        metrics = trainer.train()
        self.assertIn("best_val_acc", metrics)

        # Verify inference
        predictor = SentimentPredictor(config)
        result = predictor.predict("This AI tool is wonderful and works great!")
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn(result["label"], ["Positive", "Neutral", "Negative"])


if __name__ == "__main__":
    unittest.main()
