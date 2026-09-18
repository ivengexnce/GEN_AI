"""Dataset handling, vocabulary construction, tokenization, and sample data."""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def tokenize(text: str) -> List[str]:
    """Lowercase and extract words/tokens from string."""
    text = text.lower()
    # Simple regex word tokenizer
    tokens = re.findall(r"\b\w+\b", text)
    return tokens if tokens else [UNK_TOKEN]


def build_vocab_from_texts(texts: List[str], max_vocab_size: int = 5000) -> Dict[str, int]:
    """Build vocabulary mapping token -> integer index."""
    frequency: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            frequency[token] = frequency.get(token, 0) + 1

    # Sort tokens by frequency
    sorted_tokens = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, _ in sorted_tokens:
        if len(vocab) >= max_vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 32) -> List[int]:
    """Convert text into list of vocabulary indices padded/truncated to max_len."""
    tokens = tokenize(text)
    indices = [vocab.get(tok, vocab.get(UNK_TOKEN, UNK_IDX)) for tok in tokens]

    if len(indices) < max_len:
        indices = indices + [vocab.get(PAD_TOKEN, PAD_IDX)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return indices


class TextSentimentDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = encode_text(self.texts[index], self.vocab, self.max_len)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


def save_vocab(vocab: Dict[str, int], file_path: str) -> None:
    """Save vocabulary dictionary to JSON."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(file_path: str) -> Dict[str, int]:
    """Load vocabulary dictionary from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_training_data() -> Tuple[List[str], List[int]]:
    """Curated initial dataset for Sentiment/Tone classification."""
    samples = [
        # Negative (0)
        ("This service is terrible and keeps crashing constantly.", 0),
        ("I hate this software, it broke all my files.", 0),
        ("Worst experience ever, totally unsatisfied and frustrated.", 0),
        ("The AI model failed to produce any accurate answers.", 0),
        ("Very bad performance, sluggish and full of bugs.", 0),
        ("Customer support was completely unhelpful and rude.", 0),
        ("I am angry about this delay, unacceptable quality.", 0),
        ("System threw fatal errors and corrupted the pipeline.", 0),
        ("Disappointing results, would not recommend this tool.", 0),
        ("Terrible latency and high failure rate.", 0),
        ("Everything went wrong with this deployment.", 0),
        ("The agent got stuck in an infinite loop and failed.", 0),

        # Neutral (1)
        ("The meeting is scheduled for 3 PM tomorrow.", 1),
        ("The dataset contains five thousand records in CSV format.", 1),
        ("The file has been saved to the models directory.", 1),
        ("Python 3.11 is installed on this machine.", 1),
        ("The function takes two parameters and returns an integer.", 1),
        ("The server responded with status code 200.", 1),
        ("There are three classes in this classification task.", 1),
        ("The input text was processed without modification.", 1),
        ("Today is Friday and the office is open.", 1),
        ("The process completed in 4.2 seconds.", 1),
        ("The data pipeline reads from standard input.", 1),
        ("Running batch inference on CPU.", 1),

        # Positive (2)
        ("This AI assistant is absolutely amazing and super fast!", 2),
        ("Outstanding work, everything executed flawlessly!", 2),
        ("I love how clean and organized this codebase is.", 2),
        ("Brilliant solution, saved us hours of debugging.", 2),
        ("The model achieved ninety-nine percent accuracy, incredible!", 2),
        ("Super happy with the performance and seamless user experience.", 2),
        ("Great job! The agent completed the entire task autonomously.", 2),
        ("Excellent tool, highly recommended for all developers.", 2),
        ("The code is elegant, well-documented, and very robust.", 2),
        ("Wonderful results, couldn't be happier with this output!", 2),
        ("Top tier engineering and fantastic architecture.", 2),
        ("Very impressed by how quickly the model converged.", 2),
    ]

    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    return texts, labels
