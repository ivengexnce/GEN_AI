"""Configuration schema and loader using Pydantic and PyYAML."""
from pathlib import Path
from typing import Optional
import yaml  # type: ignore
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ModelConfig(BaseModel):
    name: str = "sentiment_classifier_mlp"
    vocab_size: int = 5000
    embedding_dim: int = 64
    hidden_dim: int = 32
    num_classes: int = 3
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 0.005
    epochs: int = 25
    model_save_dir: str = "models/saved_weights"
    model_save_name: str = "sentiment_model.pt"
    vocab_save_name: str = "vocab.json"


class DataConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    train_split: float = 0.8


class AgentConfig(BaseModel):
    name: str = "Genesis-ReAct-Agent"
    max_iterations: int = 6
    verbose: bool = True


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to default."""
    if config_path is None:
        # Default to root config/config.yaml relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        default_path = project_root / "config" / "config.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
            return AppConfig(**raw_dict)

    return AppConfig()
