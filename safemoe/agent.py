"""
Agent module for SafeMoE.
Implements tool usage, function calling formatting, and ReAct loop support.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

@dataclass
class Tool:
    """Definition of a tool available to the Agent."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    func: Callable

class AgentConfig:
    """Configuration for Agent behavior."""
    max_steps: int = 10
    stop_tokens: List[str] = None
    tool_call_start_token: str = "<tool_code>"
    tool_call_end_token: str = "</tool_code>"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.stop_tokens is None:
            self.stop_tokens = ["<|im_end|>", "</s>"]

class SafeMoEAgent:
    """
    Agent wrapper for SafeMoE models to support tool usage and reasoning.
    Supports a simplified ReAct-style loop: Thought -> Action -> Observation -> Thought.
    """
    def __init__(self, model, tokenizer, tools: List[Tool], config: AgentConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = {t.name: t for t in tools}
        self.config = config or AgentConfig()
        
        # System prompt injection for tool definition
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """Constructs the system prompt explaining available tools."""
        tool_descs = []
        for name, tool in self.tools.items():
            tool_descs.append(f"- {name}: {tool.description}\n  Parameters: {json.dumps(tool.parameters)}")
        
        prompt = (
            "You are a helpful AI assistant with access to the following tools:\n"
            + "\n".join(tool_descs) + "\n\n"
            "To use a tool, please output the following format:\n"
            "Thought: I need to use a tool because...\n"
            f"{self.config.tool_call_start_token}\n"
            "tool_name(param1=..., param2=...)\n"
            f"{self.config.tool_call_end_token}\n"
            "\nWhen you have the final answer, output 'Final Answer: ...'."
        )
        return prompt

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parses the LLM output to find tool calls.
        Looking for: <tool_code>name(args)</tool_code>
        """
        pattern = re.escape(self.config.tool_call_start_token) + r"\s*(.*?)\s*" + re.escape(self.config.tool_call_end_token)
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        
        code_str = match.group(1).strip()
        # Simple heuristic parsing: name(kwargs)
        # For a robust implementation, use `ast.parse` or a dedicated parser.
        # Here we assume python-like syntax.
        try:
            name_match = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", code_str, re.DOTALL)
            if not name_match:
                return None
            
            tool_name = name_match.group(1)
            args_str = name_match.group(2)
            
            # Very unsafe eval for demo purposes - in prod use JSON parsing or AST
            # To make it safer, we can wrap it in dict and use json if model outputs JSON
            # For now, let's assume it outputs valid python kwargs and we interpret it carefully.
            # A safer way: ask model to output JSON. Let's assume JSON for robustness in v2.
            # Upgrading heuristic: assume args_str is `key=value, key2=value2` or just eval.
            
            # Safety wrapper
            # We will use a safe evaluation context
            eval_globals = {}
            eval_locals = {}
            # Evaluate using ast.literal_eval is cleaner if structure allows, 
            # but function call syntax needs full eval.
            # Let's switch to a simplified "Extract JSON" approach if the model was trained so.
            # But adhering to the prompt "tool_name(p=v)", let's try to reconstruct args.
            
            # Placeholder for actual robust parsing logic
            # For this demo, let's just return raw string and mock execution in tests
            # Real implementation would use: json.loads(args) if we changeprompt to JSON.
            
            # Let's do a mock parse for the format: func(a=1, b="2")
            # We'll rely on the caller/model being compliant.
            
            return tool_name, args_str
            
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            return None

    def step(self, prompt: str) -> str:
        """
        Executes one step of the agent loop (single turn).
        returns the model's response.
        """
        # In a real scenario, this calls self.model.generate()
        # For now, this is a placeholder wrapper.
        pass

    def run(self, user_query: str):
        """
        Full ReAct loop execution.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Pseudo-code for the loop (since we don't have a live model instance to generate)
        # 1. format prompt
        # 2. generate
        # 3. parse tool call
        # 4. execute tool
        # 5. append observation
        # 6. repeat
        
        return messages

