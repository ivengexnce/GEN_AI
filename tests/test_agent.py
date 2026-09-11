"""Unit tests for the Agentic AI Subsystem."""
import unittest
from src.core.config import AppConfig
from src.agent.tools import Tool, ToolRegistry, calculate, get_current_time, search_knowledge, get_default_tools
from src.agent.memory import AgentMemory, AgentStep
from src.agent.react_agent import ReActAgent


class TestAgentSubsystem(unittest.TestCase):

    def test_calculator_tool(self):
        res = calculate("25 * 4 + 10")
        self.assertEqual(res, "110")

        # Test safety
        unsafe_res = calculate("__import__('os').system('dir')")
        self.assertTrue("Error" in unsafe_res or "unsupported" in unsafe_res)

    def test_knowledge_tool(self):
        res = search_knowledge("what is react?")
        self.assertIn("ReAct (Reasoning + Acting)", res)

    def test_tool_registry(self):
        registry = ToolRegistry()
        dummy_tool = Tool(
            name="Echo",
            description="Returns the input string.",
            func=lambda x: f"Echo: {x}",
        )
        registry.register(dummy_tool)
        self.assertEqual(len(registry.list_tools()), 1)
        self.assertEqual(registry.get("echo").execute("hello"), "Echo: hello")

    def test_agent_memory(self):
        memory = AgentMemory()
        memory.set_goal("Test Goal")
        step = AgentStep(
            thought="Thinking about math",
            action="Calculator",
            action_input="10 + 10",
            observation="20",
        )
        memory.add_step(step)
        trajectory = memory.get_trajectory_summary()
        self.assertIn("Test Goal", trajectory)
        self.assertIn("Thinking about math", trajectory)
        self.assertIn("Observation: 20", trajectory)

    def test_react_agent_calculation_goal(self):
        config = AppConfig()
        config.agent.verbose = False
        agent = ReActAgent(config=config)

        goal = "Calculate (100 / 4) + 15"
        final_answer = agent.run(goal)
        self.assertIsNotNone(final_answer)
        self.assertTrue(len(agent.memory.steps) > 0)
        # Check that Calculator was called and observation contains 40.0
        actions = [s.action for s in agent.memory.steps]
        self.assertIn("Calculator", actions)


if __name__ == "__main__":
    unittest.main()
