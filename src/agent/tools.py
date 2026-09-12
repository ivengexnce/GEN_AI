"""Tool definitions and registry for Agentic AI execution."""
import datetime
import math
import re
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class Tool(BaseModel):
    """Specification and callable runner for an agent tool."""
    name: str
    description: str
    func: Callable[[str], str]

    def execute(self, tool_input: str) -> str:
        """Execute the tool with error interception."""
        try:
            return str(self.func(tool_input.strip()))
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


class ToolRegistry:
    """Registry managing available tools for an AI agent."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name.lower()] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name.lower())

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def format_tool_descriptions(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- **{t.name}**: {t.description}")
        return "\n".join(lines)


# --- Built-in Tools ---

def calculate(expression: str) -> str:
    """Evaluate safe mathematical expressions."""
    cleaned = expression.replace("^", "**").strip()
    # Check for forbidden keywords that could breach safety
    if any(forbidden in cleaned.lower() for forbidden in ["__", "import", "eval", "exec", "open", "os", "sys"]):
        return "Error: Expression contains unsupported operations."

    allowed_pattern = r"^[\d\.\s\+\-\*\/\(\)\,\%\*\*\w]+$"
    if not re.match(allowed_pattern, cleaned):
        return "Error: Expression contains unsupported characters. Use numbers and standard math operators."

    try:
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "sum": sum,
            "range": range,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
        }
        result = eval(cleaned, safe_dict)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def get_current_time(_: str = "") -> str:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (UTC%z)")


def search_knowledge(query: str) -> str:
    """Simulated knowledge retrieval engine for AI and agent concepts."""
    kb = {
        "agent": (
            "An AI Agent is an autonomous system powered by an AI/LLM engine that perceives "
            "its environment, makes decisions via reasoning loops, and takes actions using tools."
        ),
        "react": (
            "ReAct (Reasoning + Acting) is a paradigm where an LLM alternates between reasoning "
            "traces ('Thought') and action execution ('Action' -> 'Observation') to solve complex multi-step problems."
        ),
        "rag": (
            "Retrieval-Augmented Generation (RAG) combines search algorithms with language models "
            "to ground responses in external dynamic or private documents."
        ),
        "transformer": (
            "The Transformer is a neural network architecture based on self-attention mechanisms, "
            "introduced by Vaswani et al. (2017), foundational to modern LLMs like GPT, Gemini, and Claude."
        ),
        "fine-tuning": (
            "Fine-tuning adapts a pre-trained model on domain-specific datasets via supervised instruction "
            "or RLHF (Reinforcement Learning from Human Feedback)."
        ),
    }

    q_lower = query.lower()
    for key, text in kb.items():
        if key in q_lower:
            return f"[Knowledge Match '{key}']: {text}"

    return f"No exact match found in knowledge base for '{query}'. Try searching for: agent, react, rag, transformer, fine-tuning."


def create_model_inference_tool() -> Tool:
    """Tool that wraps our custom-trained PyTorch Sentiment/Intent model."""
    def run_model(text: str) -> str:
        try:
            from src.model.predictor import SentimentPredictor
            predictor = SentimentPredictor()
            res = predictor.predict(text)
            return (
                f"Prediction: {res['label']} (Confidence: {res['confidence']*100:.1f}%). "
                f"Probabilities: {res['probabilities']}"
            )
        except Exception as e:
            return f"Model inference unavailable: {e}"

    return Tool(
        name="SentimentClassifier",
        description="Analyzes the tone/sentiment of input text (Negative, Neutral, Positive) using the trained PyTorch neural model.",
        func=run_model,
    )


def get_default_tools() -> ToolRegistry:
    """Initialize and populate the default tool registry."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="Calculator",
            description="Evaluates arithmetic and mathematical expressions (e.g. '(45 * 12) + 80').",
            func=calculate,
        )
    )
    registry.register(
        Tool(
            name="DateTime",
            description="Returns current system date and time. Input can be empty string.",
            func=get_current_time,
        )
    )
    registry.register(
        Tool(
            name="KnowledgeBase",
            description="Searches an indexed knowledge store for technical terms like 'agent', 'react', 'rag', 'transformer'.",
            func=search_knowledge,
        )
    )
    registry.register(create_model_inference_tool())
    return registry
