"""
Tests for Agent module.
"""
import sys
import os
import unittest
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.agent import AgentConfig, SafeMoEAgent, Tool

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Mock tool
        def weather(location: str):
            return "Sunny"
            
        self.tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            func=weather
        )
        self.agent = SafeMoEAgent(model=None, tokenizer=None, tools=[self.tool])
        
    def test_system_prompt_generation(self):
        """Test if tool info is injected into system prompt."""
        prompt = self.agent.system_prompt
        self.assertIn("get_weather", prompt)
        self.assertIn("Get current weather", prompt)
        self.assertIn("<tool_code>", prompt)
        
    def test_parse_tool_call(self):
        """Test parsing logic."""
        response = """Thought: I check weather.
<tool_code>
get_weather(location="New York")
</tool_code>
"""
        parsed = self.agent.parse_tool_call(response)
        self.assertIsNotNone(parsed)
        name, args = parsed
        self.assertEqual(name, "get_weather")
        self.assertIn("New York", args)

    def test_parse_invalid(self):
        response = "Just some text without tool code."
        parsed = self.agent.parse_tool_call(response)
        self.assertIsNone(parsed)

if __name__ == '__main__':
    unittest.main()
