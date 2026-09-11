"""Working memory, scratchpad, and state tracking for agents."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AgentStep:
    """A single reasoning and execution turn in the ReAct trajectory."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_formatted_str(self) -> str:
        lines = [f"Thought: {self.thought}"]
        if self.action:
            lines.append(f"Action: {self.action}")
        if self.action_input:
            lines.append(f"Action Input: {self.action_input}")
        if self.observation:
            lines.append(f"Observation: {self.observation}")
        return "\n".join(lines)


class AgentMemory:
    """Short-term and working memory for an agent execution run."""

    def __init__(self):
        self.goal: str = ""
        self.steps: List[AgentStep] = []
        self.final_answer: Optional[str] = None

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self.steps = []
        self.final_answer = None

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def get_trajectory_summary(self) -> str:
        """Render the complete ReAct trajectory as structured text."""
        blocks = [f"User Goal: {self.goal}\n"]
        for idx, step in enumerate(self.steps, 1):
            blocks.append(f"--- Step {idx} [{step.timestamp}] ---")
            blocks.append(step.to_formatted_str())
        if self.final_answer:
            blocks.append(f"\nFinal Answer: {self.final_answer}")
        return "\n".join(blocks)
