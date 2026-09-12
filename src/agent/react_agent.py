"""Autonomous ReAct (Reasoning + Acting) Agent Engine."""
import os
import re
from typing import Dict, Optional
from src.core.config import AppConfig, load_config
from src.agent.memory import AgentMemory, AgentStep
from src.agent.tools import ToolRegistry, get_default_tools


import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


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
            try:
                print(message)
            except UnicodeEncodeError:
                print(message.encode("ascii", errors="replace").decode("ascii"))

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
            self.log(f"🧠 Thought: {thought}")

            if is_finished:
                self.memory.final_answer = thought
                self.memory.add_step(step)
                self.log(f"\n✨ Final Answer:\n{thought}\n")
                return thought

            self.log(f"⚡ Action: {action}")
            self.log(f"📥 Action Input: {action_input}")

            # Step 2: Act / Execute Tool
            tool = self.tools.get(action) if action else None
            if not tool:
                observation = f"Error: Tool '{action}' is not registered. Available tools: {[t.name for t in self.tools.list_tools()]}"
            else:
                observation = tool.execute(action_input or "")

            step.observation = observation
            self.memory.add_step(step)
            self.log(f"👁️  Observation: {observation}\n")

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

        # Normalize word-based numbers to digits (e.g. 'twenty' -> '20')
        word_to_num = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "hundred": "100",
        }
        normalized_goal = goal
        for word, num in word_to_num.items():
            normalized_goal = re.sub(rf"\b{word}\b", num, normalized_goal, flags=re.IGNORECASE)

        # Check for user introductions: "my name is ..."
        name_match = re.search(r"my name is\s+([a-zA-Z\s]+)", goal, re.IGNORECASE)
        if name_match and not memory.steps:
            user_name = name_match.group(1).strip().title()
            return (
                f"Nice to meet you, {user_name}! I am Genesis-ReAct-Agent. I can help you train models, analyze sentiment, compute calculations, or answer AI questions. What would you like to do today?",
                None,
                None,
                True,
            )

        # Check for loop / control instructions
        if "keep running until" in goal.lower() and not memory.steps:
            return (
                "I process goals in autonomous ReAct cycles and conclude when the goal is achieved. "
                "In interactive mode, the workbench keeps running and is ready for your next command.",
                None,
                None,
                True,
            )

        # Check for special math pattern: "sum of first N numbers" or "sum and product"
        sum_prod_match = re.search(
            r"sum\s+and\s+product\s+of\s+(?:the\s+)?first\s+(\d+)\s*(whole|natural)?\s*numbers",
            normalized_goal,
            re.IGNORECASE,
        )
        sum_match = re.search(
            r"sum\s+of\s+(?:the\s+)?first\s+(\d+)\s*(whole|natural)?\s*numbers",
            normalized_goal,
            re.IGNORECASE,
        )

        math_done = "calculator" in executed_actions

        if sum_prod_match and not math_done:
            n = int(sum_prod_match.group(1))
            num_type = (sum_prod_match.group(2) or "whole").lower()
            if num_type == "whole":
                # Whole numbers start at 0 (0 to n-1)
                sum_val = sum(range(n))
                prod_val = 0
                return (
                    f"Calculated for first {n} whole numbers (0 to {n-1}): "
                    f"Sum = sum(range({n})) = {sum_val}. Product = 0 (since whole numbers include 0, anything times 0 is 0).",
                    None,
                    None,
                    True,
                )
            else:
                sum_val = sum(range(1, n + 1))
                return (
                    f"The user requested sum and product. I will use the Calculator tool to compute the sum.",
                    "Calculator",
                    f"sum(range(1, {n + 1}))",
                    False,
                )

        if sum_match and not math_done:
            n = int(sum_match.group(1))
            num_type = (sum_match.group(2) or "whole").lower()
            expr = f"sum(range({n}))" if num_type == "whole" else f"sum(range(1, {n + 1}))"
            return (
                f"The user requested the sum of the first {n} {num_type} numbers. I will use the Calculator to evaluate '{expr}'.",
                "Calculator",
                expr,
                False,
            )

        # Check for Math/Calculation needs
        math_match = re.search(r"([\d\.\s\+\-\*\/\(\)\^\,]{3,})", normalized_goal)
        has_math = bool(math_match and any(op in normalized_goal.lower() for op in ["+", "-", "*", "/", "^", "calculate", "sum", "multiply", "product"]))

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
            expr = math_match.group(1).strip() if math_match else normalized_goal
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

        # Check for conversational queries, greetings, or direct speech requests
        clean_goal = goal.strip()
        lower_goal = clean_goal.lower()

        # Greetings
        if any(lower_goal == g or lower_goal.startswith(f"{g} ") or lower_goal.endswith(f" {g}") for g in ["hello", "hi", "hey", "greetings", "good morning", "good evening", "say hello"]):
            return (
                "Hello! I am your Genesis ReAct AI Agent. I can assist you with running PyTorch model training, classifying sentiment, calculating math expressions, querying domain knowledge, or checking the current date and time. How can I help you today?",
                None,
                None,
                True,
            )

        # Farewells
        if any(lower_goal == b or lower_goal.startswith(f"{b} ") for b in ["bye", "goodbye", "say bye", "exit", "quit"]):
            return (
                "Goodbye! Feel free to return anytime to test models or run autonomous agents.",
                None,
                None,
                True,
            )

        # "Say <text>" requests
        if lower_goal.startswith("say "):
            phrase = clean_goal[4:].strip()
            return (
                f"{phrase}",
                None,
                None,
                True,
            )

        # Agent Identity / Help
        if any(k in lower_goal for k in ["who are you", "what are you", "what can you do", "help"]):
            tool_names = ", ".join([t.name for t in self.tools.list_tools()])
            return (
                f"I am {self.config.agent.name}, an autonomous AI agent built on the ReAct (Reasoning + Acting) paradigm. "
                f"I can autonomously reason about tasks and execute tools ({tool_names}) or LLM endpoints to achieve goals.",
                None,
                None,
                True,
            )

        # Fallback if no specific trigger matched
        return (
            f"I have reviewed the goal '{goal}'. No external tool execution was required or the query was self-contained.",
            None,
            None,
            True,
        )

    def _plan_with_gemini(self, goal: str, memory: AgentMemory, api_key: str):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError("Package `google-generativeai` is required. Run `pip install google-generativeai`.") from e
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
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("Package `openai` is required. Run `pip install openai`.") from e
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
