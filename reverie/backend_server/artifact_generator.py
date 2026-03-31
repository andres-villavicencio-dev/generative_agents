"""
Artifact Content Generation — Phase 2

Generates rich, full-text content for artifacts created by agents.
Each artifact type gets a specialized LLM prompt that produces
appropriate content (prose, lyrics, image prompts, etc.).
"""

import os
import re
import json
import traceback
from datetime import datetime


def _sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower()).strip('_')


def _get_artifacts_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _save_to_file(subdir, agent_name, content):
    """Save content to artifacts/{subdir}/{agent}_{timestamp}.txt. Returns path."""
    base = os.path.join(_get_artifacts_dir(), subdir)
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_sanitize_filename(agent_name)}_{ts}.txt"
    path = os.path.join(base, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


def _append_log(agent, artifact_type, title, path, content):
    log_path = os.path.join(_get_artifacts_dir(), "artifact_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "type": artifact_type,
        "title": title,
        "path": path,
        "preview": content[:100] if content else "",
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _llm_call(prompt):
    """Call LLM with free_form=True for unconstrained text generation."""
    from persona.prompt_template.gpt_structure import (
        USE_LLAMA_CPP, _llama_cpp_generate, _ollama_generate
    )
    if USE_LLAMA_CPP:
        return _llama_cpp_generate(prompt, free_form=True)
    else:
        return _ollama_generate(prompt, free_form=True)


def generate_written_content(agent_name, action, persona_description):
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Write the full text of what {agent_name} wrote. "
        f"200-400 words. Write in {agent_name}'s voice and personality. "
        f"No meta-commentary, no titles, just the written content itself."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("writings", agent_name, content)
            _append_log(agent_name, "writing", action, path, content)
            return content
    except Exception:
        traceback.print_exc()
    return ""


def generate_painting_prompt(agent_name, action, persona_description):
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Describe this artwork as a detailed image generation prompt. "
        f"Include: subject, artistic style, color palette, mood, composition, "
        f"lighting. 2-4 sentences. Output only the image prompt."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("paintings", agent_name, content)
            _append_log(agent_name, "painting", action, path, content)
            
            image_result = None
            try:
                image_result = _generate_painting_image(content, agent_name)
            except Exception as img_e:
                print(f"[ArtifactGenerator] Image generation skipped: {img_e}")
            
            if image_result:
                return json.dumps({
                    "prompt": content,
                    "image_path": image_result.get("image_path"),
                    "model": image_result.get("model", "unknown")
                })
            
            return content
    except Exception:
        traceback.print_exc()
    return ""


def _generate_painting_image(prompt, agent_name):
    """Attempt to generate an actual image from the prompt."""
    try:
        import subprocess
        image_dir = os.path.join(_get_artifacts_dir(), "images")
        os.makedirs(image_dir, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', agent_name.lower())
        output_path = os.path.join(image_dir, f"{safe_name}_{ts}.png")
        
        return {
            "image_path": output_path,
            "model": "placeholder",
            "note": "Image generation requires Stable Diffusion setup"
        }
    except Exception as e:
        return None


def generate_painting_with_image(agent_name, action, persona_description):
    """Generate painting prompt and call ImageGeneratorTool.
    
    Returns a JSON string with prompt and optional image_path.
    """
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Describe this artwork as a detailed image generation prompt. "
        f"Include: subject, artistic style, color palette, mood, composition, "
        f"lighting. 2-4 sentences. Output only the image prompt."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("paintings", agent_name, content)
            
            result = {
                "prompt": content,
                "prompt_file": path,
                "image_path": None
            }
            
            image_result = None
            try:
                from tool_registry import ImageGeneratorTool
                tool = ImageGeneratorTool()
                class MockPersona:
                    scratch = type('scratch', (), {
                        'act_description': action,
                        'name': agent_name,
                        'get_str_iss': lambda: persona_description
                    })()
                image_result = tool._call_image_model(content, agent_name)
            except Exception as img_e:
                print(f"[ArtifactGenerator] Image generation skipped: {img_e}")
            
            if image_result and image_result.get('success'):
                result['image_path'] = image_result.get('image_path')
                result['model'] = image_result.get('model', 'unknown')
            
            _append_log(agent_name, "painting", action, path, json.dumps(result))
            return json.dumps(result)
    except Exception:
        traceback.print_exc()
    return ""


def generate_song_lyrics(agent_name, action, persona_description):
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Write the complete song lyrics that {agent_name} composed. "
        f"Include verses and a chorus. 100-300 words. "
        f"Write in {agent_name}'s voice. No meta-commentary."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("songs", agent_name, content)
            _append_log(agent_name, "song", action, path, content)
            return content
    except Exception:
        traceback.print_exc()
    return ""


def generate_meal_description(agent_name, action, persona_description):
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Describe the meal {agent_name} prepared: dish name, key ingredients, "
        f"preparation method, aroma, presentation, and the occasion. "
        f"100-200 words."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("meals", agent_name, content)
            _append_log(agent_name, "meal", action, path, content)
            return content
    except Exception:
        traceback.print_exc()
    return ""


def generate_invention_description(agent_name, action, persona_description):
    prompt = (
        f"{agent_name} is {action}.\n"
        f"About {agent_name}: {persona_description}\n\n"
        f"Describe what {agent_name} built or invented: what it does, "
        f"how it works, materials used, and why they made it. "
        f"100-200 words."
    )
    try:
        content = _llm_call(prompt)
        if content:
            path = _save_to_file("inventions", agent_name, content)
            _append_log(agent_name, "invention", action, path, content)
            return content
    except Exception:
        traceback.print_exc()
    return ""


# Keyword → (generator_function, artifact_log_type)
_CONTENT_DISPATCH = [
    (["build", "invent", "craft"],
     generate_invention_description, "invention"),
    (["cook", "bake", "meal", "food", "recipe"],
     generate_meal_description, "meal"),
    (["paint", "draw", "sketch", "artwork"],
     generate_painting_with_image, "painting"),
    (["song", "music", "compos", "sing"],
     generate_song_lyrics, "song"),
    (["writing", "letter", "essay", "poem", "journal", "book"],
     generate_written_content, "writing"),
]


def generate_artifact_content(agent_name, action, persona_description):
    """Dispatch to the appropriate generator based on action keywords.
    Returns (content_full, content_type) or ("", None) if no match."""
    action_lower = action.lower()
    for keywords, generator_fn, content_type in _CONTENT_DISPATCH:
        for kw in keywords:
            if kw in action_lower:
                content = generator_fn(agent_name, action, persona_description)
                return content, content_type
    print(f"[ArtifactGenerator] No content type matched for action: {action}")
    return "", None
