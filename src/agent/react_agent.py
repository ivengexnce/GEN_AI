"""Autonomous ReAct (Reasoning + Acting) Agent Engine."""
import os
import re
from typing import Dict, Optional
from src.core.config import AppConfig, load_config
from src.agent.memory import AgentMemory, AgentStep
from src.agent.tools import ToolRegistry, get_default_tools


class ReActAgent:
    """
    Autonomous AI Agent implementing the ReAct framework:
    Thought -> Action -> Action Input -> Observation -> Final Answer.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.config = config or load_config()
        self.tools = tool_registry or get_default_tools()
        self.memory = AgentMemory()
        self.max_iterations = self.config.agent.max_iterations
        self.verbose = self.config.agent.verbose

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run(self, goal: str) -> str:
        """Run the autonomous reasoning and action loop to achieve the given goal."""
        self.memory.set_goal(goal)
        self.log(f"\n=======================================================")
        self.log(f"  [AGENT] Started: {self.config.agent.name}")
        self.log(f"  [GOAL]  {goal}")
        self.log(f"=======================================================\n")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.log(f"--- [Cycle {iteration}/{self.max_iterations}] ---")

            # Step 1: Reason / Plan next action given history
            thought, action, action_input, is_finished = self._plan_step(goal, self.memory)

            step = AgentStep(thought=thought, action=action, action_input=action_input)
            self.log(f"[Thought] {thought}")

            if is_finished:
                self.memory.final_answer = thought
                self.memory.add_step(step)
                self.log(f"\n[Final Answer]\n{thought}\n")
                return thought

            self.log(f"[Action] {action}")
            self.log(f"[Action Input] {action_input}")

            # Step 2: Act / Execute Tool
            tool = self.tools.get(action) if action else None
            if not tool:
                observation = f"Error: Tool '{action}' is not registered. Available tools: {[t.name for t in self.tools.list_tools()]}"
            else:
                observation = tool.execute(action_input or "")

            step.observation = observation
            self.memory.add_step(step)
            self.log(f"[Observation] {observation}\n")

        fallback = (
            f"Agent reached maximum limit of {self.max_iterations} iterations without concluding. "
            f"Partial context:\n{self.memory.get_trajectory_summary()}"
        )
        self.memory.final_answer = fallback
        return fallback

    def _plan_step(
        self, goal: str, memory: AgentMemory
    ) -> tuple[str, Optional[str], Optional[str], bool]:
        """
        Determine the next Thought, Action, and Action Input.
        Supports both LLM (if API key is available) and robust semantic-heuristic decomposition.
        """
        # Check if an LLM key is set
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key:
            try:
                return self._plan_with_gemini(goal, memory, gemini_key)
            except Exception as e:
                self.log(f"[*] Gemini API call skipped ({e}), using autonomous reasoning engine.")
        elif openai_key:
            try:
                return self._plan_with_openai(goal, memory, openai_key)
            except Exception as e:
                self.log(f"[*] OpenAI API call skipped ({e}), using autonomous reasoning engine.")

        # Built-in Autonomous Reasoning Engine (zero external dependencies/keys needed)
        return self._autonomous_heuristic_reasoning(goal, memory)

    def _autonomous_heuristic_reasoning(
        self, goal: str, memory: AgentMemory
    ) -> tuple[str, Optional[str], Optional[str], bool]:
        """
        Inspect goal and previous observations to chain tools logically.
        """
        executed_actions = [s.action.lower() for s in memory.steps if s.action]
        last_step = memory.steps[-1] if memory.steps else None

        # Check for Math/Calculation needs
        math_match = re.search(r"([\d\.\s\+\-\*\/\(\)\^]{3,})", goal)
        has_math = bool(math_match and any(op in goal for op in ["+", "-", "*", "/", "^", "calculate", "sum", "multiply"]))
        math_done = "calculator" in executed_actions

        # Check for Sentiment/Tone analysis needs
        sentiment_keywords = ["sentiment", "analyze", "tone", "emotion", "feeling", "review"]
        has_sentiment = any(k in goal.lower() for k in sentiment_keywords)
        sentiment_done = "sentimentclassifier" in executed_actions

        # Check for Knowledge Base needs
        knowledge_keywords = ["what is", "explain", "definition", "define", "concept", "knowledge", "rag", "react", "transformer"]
        has_knowledge = any(k in goal.lower() for k in knowledge_keywords)
        knowledge_done = "knowledgebase" in executed_actions

        # Check for Time needs
        time_keywords = ["time", "date", "clock", "today"]
        has_time = any(k in goal.lower() for k in time_keywords)
        time_done = "datetime" in executed_actions

        # Decision 1: Do we need calculation?
        if has_math and not math_done:
            expr = math_match.group(1).strip() if math_match else goal
            # Clean non-math words
            expr_clean = re.sub(r"[a-zA-Z]", "", expr).strip()
            return (
                "The user requested a calculation. I will use the Calculator tool to compute the mathematical expression.",
                "Calculator",
                expr_clean or expr,
                False
            )

        # Decision 2: Do we need sentiment analysis?
        if has_sentiment and not sentiment_done:
            # Extract quoted text or relevant clause
            quote_match = re.search(r"['\"](.*?)['\"]", goal)
            target_text = quote_match.group(1) if quote_match else goal
            return (
                f"The user wants text sentiment or tone classified. I will invoke the SentimentClassifier neural model.",
                "SentimentClassifier",
                target_text,
                False
            )

        # Decision 3: Do we need knowledge lookup?
        if has_knowledge and not knowledge_done:
            # Extract topic
            query = re.sub(r"(what is|explain|define|tell me about)", "", goal, flags=re.IGNORECASE).strip()
            return (
                f"The goal requires technical domain knowledge. I will search the KnowledgeBase for '{query}'.",
                "KnowledgeBase",
                query or goal,
                False
            )

        # Decision 4: Do we need date/time?
        if has_time and not time_done:
            return (
                "The goal requires current time/date information. I will query the DateTime tool.",
                "DateTime",
                "",
                False
            )

        # Decision 5: Synthesize and conclude
        if memory.steps:
            collected = []
            for i, step in enumerate(memory.steps, 1):
                collected.append(f"- Step {i} ({step.action}): {step.observation}")
            summary = "\n".join(collected)
            thought = (
                f"I have gathered all required observations to fulfill the goal '{goal}'.\n\n"
                f"Results summary:\n{summary}"
            )
            return thought, None, None, True

        # Fallback if no specific trigger matched
        return (
            f"I have reviewed the goal '{goal}'. No external tool execution was required or the query was self-contained.",
            None,
            None,
            True
        )

    def _plan_with_gemini(self, goal: str, memory: AgentMemory, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        tools_doc = self.tools.format_tool_descriptions()
        trajectory = memory.get_trajectory_summary()

        prompt = f"""You are a ReAct AI agent.
Available Tools:
{tools_doc}

Format your response strictly as:
Thought: <reasoning>
Action: <tool_name or None>
Action Input: <input string or None>
Final Answer: <final response if finished, otherwise leave blank>

Current Trajectory:
{trajectory}

Next Step:"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        return self._parse_react_response(text)

    def _plan_with_openai(self, goal: str, memory: AgentMemory, api_key: str):
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        tools_doc = self.tools.format_tool_descriptions()
        trajectory = memory.get_trajectory_summary()

        prompt = f"""You are a ReAct AI agent.
Available Tools:
{tools_doc}

Format your response strictly as:
Thought: <reasoning>
Action: <tool_name or None>
Action Input: <input string or None>
Final Answer: <final response if finished, otherwise leave blank>

Current Trajectory:
{trajectory}

Next Step:"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = res.choices[0].message.content.strip()
        return self._parse_react_response(text)

    def _parse_react_response(self, text: str) -> tuple[str, Optional[str], Optional[str], bool]:
        if "Final Answer:" in text:
            final_ans = text.split("Final Answer:", 1)[1].strip()
            return final_ans, None, None, True

        thought = "Analyzing trajectory..."
        action = None
        action_input = None

        if "Thought:" in text:
            thought = text.split("Thought:")[1].split("Action:")[0].strip()
        if "Action:" in text:
            action = text.split("Action:")[1].split("Action Input:")[0].strip()
        if "Action Input:" in text:
            action_input = text.split("Action Input:")[1].split("\n")[0].strip()

        return thought, action, action_input, False
