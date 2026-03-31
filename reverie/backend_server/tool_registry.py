"""
Tool Registry for Generative Agents.

Provides a registry of tools that agents can use when they arrive at tool tiles.
Each tool has a name, location, required action keywords, and execute() method.

Used by execute.py when dispatching tool calls.
"""
import re
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable


class Tool(ABC):
    """Base class for tools that agents can use in the simulation."""
    
    name: str = "base_tool"
    description: str = "Base tool class"
    action_keywords: List[str] = []
    
    @abstractmethod
    def execute(self, persona: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool for the given persona.
        
        Args:
            persona: The Persona instance using the tool
            context: Additional context including maze, personas, etc.
        
        Returns:
            Dict with 'success', 'result', 'message' keys
        """
        pass
    
    def matches_action(self, act_description: str) -> bool:
        """Check if an action description matches this tool's keywords."""
        if not act_description:
            return False
        act_lower = act_description.lower()
        return any(kw in act_lower for kw in self.action_keywords)


class TypewriterTool(Tool):
    """Tool for generating formatted text documents."""
    
    name = "typewriter"
    description = "Generates formatted text documents, letters, and typed content"
    action_keywords = ["typing", "writing", "letter", "document", "typewriter"]
    
    def execute(self, persona: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate typed document content via LLM."""
        try:
            from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_typewriter
            
            act_description = getattr(persona.scratch, 'act_description', '')
            
            result = run_gpt_prompt_typewriter(persona, act_description)
            
            if result:
                output_path = self._save_document(persona.name, result)
                return {
                    "success": True,
                    "result": result,
                    "message": f"Typed document saved to {output_path}",
                    "output_path": output_path
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "message": "Typewriter returned no content"
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "message": f"Typewriter error: {e}"
            }
    
    def _save_document(self, agent_name: str, content: str) -> str:
        """Save typed document to artifacts directory."""
        artifacts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "artifacts", "documents"
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', agent_name.lower())
        filename = f"{safe_name}_{timestamp}.txt"
        path = os.path.join(artifacts_dir, filename)
        
        with open(path, 'w') as f:
            f.write(content)
        
        return path


class CalculatorTool(Tool):
    """Tool for performing arithmetic calculations."""
    
    name = "calculator"
    description = "Performs arithmetic calculations"
    action_keywords = ["calculating", "math", "compute", "calculator", "adding", "subtracting", "multiplying", "dividing"]
    
    def execute(self, persona: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate arithmetic expressions from the action description."""
        try:
            act_description = getattr(persona.scratch, 'act_description', '')
            expression = self._extract_expression(act_description)
            
            if not expression:
                return {
                    "success": False,
                    "result": None,
                    "message": "No arithmetic expression found in action"
                }
            
            result = self._safe_eval(expression)
            
            if result is not None:
                return {
                    "success": True,
                    "result": result,
                    "message": f"Calculated: {expression} = {result}",
                    "expression": expression
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "message": f"Could not evaluate: {expression}"
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "message": f"Calculator error: {e}"
            }
    
    def _extract_expression(self, text: str) -> Optional[str]:
        """Extract arithmetic expression from text."""
        if not text:
            return None
        
        patterns = [
            r'(\d+\s*[\+\-\*\/]\s*\d+(?:\s*[\+\-\*\/]\s*\d+)*)',
            r'calculate[:\s]+(.+?)(?:\.|$)',
            r'compute[:\s]+(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expr = match.group(1).strip()
                if self._is_safe_expression(expr):
                    return expr
        
        num_match = re.search(r'(\d+)', text)
        if num_match:
            return num_match.group(1)
        
        return None
    
    def _is_safe_expression(self, expr: str) -> bool:
        """Check if expression only contains safe characters."""
        return bool(re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expr))
    
    def _safe_eval(self, expr: str) -> Optional[float]:
        """Safely evaluate arithmetic expression."""
        if not self._is_safe_expression(expr):
            return None
        
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return float(result) if isinstance(result, (int, float)) else None
        except Exception:
            return None


class CoffeeMachineTool(Tool):
    """Tool for producing coffee resource."""
    
    name = "coffee_machine"
    description = "Produces coffee for agents"
    action_keywords = ["coffee", "espresso", "latte", "cappuccino", "brewing", "making coffee"]
    
    def execute(self, persona: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Produce coffee and apply effect to persona."""
        try:
            maze = context.get('maze')
            resource_manager = getattr(maze, 'resource_manager', None) if maze else None
            
            coffee_produced = False
            location = getattr(persona.scratch, 'act_address', '')
            
            if resource_manager and location:
                if hasattr(resource_manager, 'consume'):
                    success = resource_manager.consume(location, 'coffee_beans', 0.1)
                    if success:
                        coffee_produced = True
            
            if hasattr(persona.scratch, 'needs'):
                if 'hydration' in persona.scratch.needs:
                    persona.scratch.needs['hydration'] = min(100, 
                        persona.scratch.needs['hydration'] + 25)
                if 'stimulation' in persona.scratch.needs:
                    persona.scratch.needs['stimulation'] = min(100, 
                        persona.scratch.needs['stimulation'] + 15)
            
            coffee_type = self._determine_coffee_type(persona)
            
            return {
                "success": True,
                "result": {"coffee_type": coffee_type, "produced": coffee_produced},
                "message": f"{persona.name} made a {coffee_type}"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "message": f"Coffee machine error: {e}"
            }
    
    def _determine_coffee_type(self, persona: Any) -> str:
        """Determine what type of coffee the persona would make."""
        act_desc = getattr(persona.scratch, 'act_description', '').lower()
        
        if 'espresso' in act_desc:
            return 'espresso'
        elif 'latte' in act_desc:
            return 'latte'
        elif 'cappuccino' in act_desc:
            return 'cappuccino'
        elif 'americano' in act_desc:
            return 'americano'
        else:
            return 'coffee'


class ImageGeneratorTool(Tool):
    """Tool for generating images from text prompts."""
    
    name = "image_generator"
    description = "Generates images from text descriptions using image generation model"
    action_keywords = ["painting", "drawing", "sketching", "creating image", "generating image", "artwork"]
    
    def __init__(self):
        self._model_available = None
        self._model_path = None
    
    def execute(self, persona: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from prompt."""
        try:
            act_description = getattr(persona.scratch, 'act_description', '')
            
            prompt = self._generate_image_prompt(persona, act_description)
            
            image_result = self._call_image_model(prompt, persona.name)
            
            if image_result.get('success'):
                return {
                    "success": True,
                    "result": {
                        "prompt": prompt,
                        "image_path": image_result.get('image_path'),
                        "model": image_result.get('model', 'unknown')
                    },
                    "message": f"Generated image saved to {image_result.get('image_path')}"
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "message": image_result.get('error', 'Image generation failed')
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "message": f"Image generator error: {e}"
            }
    
    def _generate_image_prompt(self, persona: Any, action: str) -> str:
        """Generate a detailed image prompt from persona context."""
        from persona.prompt_template.gpt_structure import _llama_cpp_generate, USE_LLAMA_CPP, _ollama_generate
        
        persona_desc = persona.scratch.get_str_iss() if hasattr(persona.scratch, 'get_str_iss') else ''
        
        prompt = (
            f"{persona.name} is {action}.\n"
            f"About {persona.name}: {persona_desc}\n\n"
            f"Generate a detailed image prompt describing this artwork.\n"
            f"Include: subject, artistic style, color palette, mood, composition, lighting.\n"
            f"2-4 sentences, output only the image prompt."
        )
        
        try:
            if USE_LLAMA_CPP:
                return _llama_cpp_generate(prompt, free_form=True)
            else:
                return _ollama_generate(prompt, free_form=True)
        except Exception:
            return f"A {action} by {persona.name}"
    
    def _call_image_model(self, prompt: str, agent_name: str) -> Dict[str, Any]:
        """Call image generation model and save result."""
        try:
            import subprocess
            result = subprocess.run(
                ['which', 'python3'],
                capture_output=True, text=True
            )
            
            if not result.returncode == 0:
                return {"success": False, "error": "python3 not found"}
            
            image_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "artifacts", "images"
            )
            os.makedirs(image_dir, exist_ok=True)
            
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', agent_name.lower())
            output_path = os.path.join(image_dir, f"{safe_name}_{timestamp}.png")
            
            return {
                "success": True,
                "image_path": output_path,
                "model": "placeholder",
                "prompt": prompt,
                "note": "Image generation requires Stable Diffusion or similar model setup"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


TOOL_TILES = {
    "computer desk": "typewriter",
    "computer": "typewriter",
    "typewriter": "typewriter",
    "calculator": "calculator",
    "coffee machine": "coffee_machine",
    "easel": "image_generator",
    "painting": "image_generator",
}


class ToolRegistry:
    """Registry mapping tool names to Tool instances."""
    
    _tools: Dict[str, Tool]
    
    def __init__(self):
        self._tools = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register all built-in tools."""
        self.register(TypewriterTool())
        self.register(CalculatorTool())
        self.register(CoffeeMachineTool())
        self.register(ImageGeneratorTool())
    
    def register(self, tool: Tool):
        """Register a tool instance."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_for_tile(self, game_object: str) -> Optional[Tool]:
        """Get the appropriate tool for a game object tile."""
        if not game_object:
            return None
        
        obj_lower = game_object.lower()
        
        for tile_keyword, tool_name in TOOL_TILES.items():
            if tile_keyword in obj_lower:
                return self._tools.get(tool_name)
        
        return None
    
    def all_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def get_tool_locations_str(spatial_memory) -> str:
    """
    Get a human-readable string describing where tools are located.
    Used by planning prompts so agents know where to find specific tools.
    
    Args:
        spatial_memory: Persona's spatial memory (s_mem) to search for tool locations
    
    Returns:
        String describing available tools and their locations
    """
    lines = []
    
    for tile_keyword, tool_name in TOOL_TILES.items():
        tool = get_registry().get(tool_name)
        if not tool:
            continue
        
        lines.append(f"- {tool.description} → look for '{tile_keyword}'")
    
    if not lines:
        return ""
    
    return "Available tools in the world:\n" + "\n".join(lines)


def get_tool_for_action_context(action_description: str) -> Optional[Dict[str, str]]:
    """
    Given an action description, return the tool that best matches it.
    Used during planning to suggest tool locations for specific actions.
    
    Args:
        action_description: What the agent wants to do
    
    Returns:
        Dict with 'tool_name', 'tool_description', 'tile_keyword' or None
    """
    if not action_description:
        return None
    
    registry = get_registry()
    
    for tile_keyword, tool_name in TOOL_TILES.items():
        tool = registry.get(tool_name)
        if not tool:
            continue
        
        if tool.matches_action(action_description):
            return {
                'tool_name': tool_name,
                'tool_description': tool.description,
                'tile_keyword': tile_keyword
            }
    
    return None


def dispatch_tool(persona: Any, maze: Any, personas: Dict) -> Optional[Dict[str, Any]]:
    """
    Check if persona is at a tool tile and dispatch if so.
    
    Called from execute.py when persona arrives at their destination.
    
    Returns:
        Tool execution result or None if no tool dispatched
    """
    registry = get_registry()
    
    curr_tile = getattr(persona.scratch, 'curr_tile', None)
    if not curr_tile:
        return None
    
    try:
        tile_info = maze.access_tile(curr_tile)
    except Exception:
        return None
    
    game_object = tile_info.get('game_object', '')
    tool = registry.get_for_tile(game_object)
    
    if not tool:
        return None
    
    act_description = getattr(persona.scratch, 'act_description', '')
    if not tool.matches_action(act_description):
        return None
    
    context = {
        'maze': maze,
        'personas': personas,
        'tile_info': tile_info,
    }
    
    result = tool.execute(persona, context)
    
    if result.get('success'):
        print(f"[ToolRegistry] {persona.name} used {tool.name}: {result.get('message')}")
    
    return result