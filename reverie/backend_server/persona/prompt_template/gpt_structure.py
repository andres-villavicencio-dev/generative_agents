"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified: Local Ollama adaptation

File: gpt_structure.py
Description: Wrapper functions for calling local Ollama APIs instead of OpenAI.
"""
# =============================================================================
# LLM Backend Configuration
# =============================================================================
# This module supports two backends for text generation:
#
# 1. Ollama (default): Uses /api/generate endpoint with JSON schema constraints.
#    Set USE_LLAMA_CPP = False (default).
#
# 2. llama.cpp server: Uses /completion endpoint with GBNF grammar constraints.
#    Set USE_LLAMA_CPP = True and ensure llama.cpp server is running.
#    See llama_cpp_server/README.md for setup instructions.
#
# TurboQuant Integration Note:
#   The turboquant/ package provides KV cache quantization that can be used
#   alongside either backend. The llama.cpp server also supports its own
#   built-in KV cache quantization via --cache-type-k and --cache-type-v flags.
# =============================================================================
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

from utils import *

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("GA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("GA_CHAT_MODEL", "qwen3.5:2b")

# ---------------------------------------------------------------------------
# Multi-model routing: GA's call families have wildly different shapes.
#   fast lane   (~85% of calls): short, format-strict, high-frequency
#   heavy lanes (rare, quality-critical): planning / artifact / convo / insight
# Models are selected per call via model_hint, resolved through env vars so
# each lane is independently swappable WITHOUT code changes:
#   GA_PLANNING_MODEL, GA_ARTIFACT_MODEL, GA_CONVO_MODEL, GA_INSIGHT_MODEL
# Probed 2026-09-23 (see session log): every 2025-26 reasoning-first sLLM
# (qwen3.5:2b, ornith-1.5:9b, lfm2.5:8b-a1b) breaks GA's 2023-era completion-
# style prompt library. gemma3:latest is the only probe-clean model per lane.
# Keep defaults == OLLAMA_CHAT_MODEL until a candidate passes probe.
# Per-model context windows: gemma3=131k but KV-heavy -> cap 8192;
# future non-reasoning models may want larger ctx via this table.
# ---------------------------------------------------------------------------
OLLAMA_MODEL_ROUTES = {
    "planning": os.environ.get("GA_PLANNING_MODEL", OLLAMA_CHAT_MODEL),
    "artifact": os.environ.get("GA_ARTIFACT_MODEL", OLLAMA_CHAT_MODEL),
    "convo":    os.environ.get("GA_CONVO_MODEL",    OLLAMA_CHAT_MODEL),
    "insight":  os.environ.get("GA_INSIGHT_MODEL",  OLLAMA_CHAT_MODEL),
}
_NUM_CTX_BY_MODEL = {
    "gemma3:latest": 8192,
    # LFM2.5-24B-A2B thinking lane (Oct 15 2026): reasoning tokens + long
    # persona prompts need ctx headroom; the conv-hybrid arch (30 gated
    # short-conv + 10 GQA layers) carries a tiny KV, so 16384 is cheap.
    "hf.co/LiquidAI/LFM2.5-24B-A2B-GGUF:Q4_K_M": 16384,
    # default 8192 keeps KV VRAM in check on the 8GB card
}

def _resolve_model(model_hint):
    """Return (model_name, num_ctx) for a call-family hint (or None for fast lane)."""
    model = OLLAMA_MODEL_ROUTES.get(model_hint, OLLAMA_CHAT_MODEL)
    return model, _NUM_CTX_BY_MODEL.get(model, 8192)


# --- Per-persona generation settings (Layer 1, Oct 2026) ------------------
# A move-worker thread handles ONE agent's whole move(), so a thread-local
# carries call attribution without touching the call signatures. Set in
# reverie.py's _move_one; read here to override temperature/think per persona.
_PERSONA_SETTINGS = {}       # name -> settings dict (loaded once, lazily)
_PERSONA_CTX = threading.local()   # .name set by _move_one for this thread


def set_persona_call_context(name):
    """reverie.py calls this at the top of _move_one to attribute all LLM
    calls on this worker thread to the given persona."""
    _PERSONA_CTX.name = name


def _persona_settings():
    """Return this thread's persona settings dict (or {}). Loads
    town_model_settings.json once; missing names/keys are absent."""
    name = getattr(_PERSONA_CTX, "name", None)
    if not name:
        return {}
    if "loaded" not in _PERSONA_SETTINGS:
        try:
            import json as _json
            _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "town_model_settings.json")
            with open(_path) as f:
                data = _json.load(f)
            _PERSONA_SETTINGS.update({k: v for k, v in data.items()
                                      if not k.startswith("_")})
            _PERSONA_SETTINGS["loaded"] = True
            print(f"[ModelSettings] loaded {len(data)-1} persona settings")
        except Exception as e:
            _PERSONA_SETTINGS["loaded"] = True
            print(f"[ModelSettings] settings load failed ({e}); defaults apply")
    return _PERSONA_SETTINGS.get(name) or {}


def _persona_temperature(default):
    """Persona temperature override, clamped 0.1-1.2."""
    t = _persona_settings().get("temperature")
    if t is None:
        return default
    return max(0.1, min(1.2, float(t)))


def _persona_think(lane_think):
    """Per-persona reasoning override (True/False), or the lane default."""
    p = _persona_settings().get("think")
    if p is None:
        return lane_think
    return bool(p)


_LANE_THINK_ENV = {
    None:      "GA_FAST_THINK",
    "planning": "GA_PLANNING_THINK",
    "artifact": "GA_ARTIFACT_THINK",
    "convo":    "GA_CONVO_THINK",
    "insight":  "GA_INSIGHT_THINK",
}

def _lane_think(model_hint):
    """Per-lane reasoning toggle for /api/chat. Defaults to False: GA call
    formats need the answer at position 1, and most of the volume is
    short-format calls where reasoning tokens are pure latency. Lanes may
    opt into think=true (quality over speed) via GA_<LANE>_THINK=1."""
    env_key = _LANE_THINK_ENV.get(model_hint, "GA_FAST_THINK")
    return os.environ.get(env_key, "0") == "1"
OLLAMA_EMBED_MODEL = "embeddinggemma:latest"

# llama.cpp server configuration
LLAMA_CPP_BASE_URL = "http://localhost:8080"
USE_LLAMA_CPP = False  # Set True to use llama.cpp instead of Ollama

# GBNF grammar for JSON schema constraint: {"output": "<string>"}
_GBNF_JSON_OUTPUT = r'''root   ::= "{" ws "\"output\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws     ::= [ \t\n]*'''

# YAML lane (Oct 2026): qwen3.5-family models drift less with a minimal YAML
# shape than with JSON objects (A/B probe Oct 10: 12/12 vs 10/12 on the event
# triple). Grammar-constrain the output so YAML is as hard a guarantee as the
# old format:"json" — a single "output: <value>" line.
_GBNF_YAML_OUTPUT = r'''root   ::= "output: " value
value ::= [^\n]+'''

# _parse_yaml_output: stdlib-only parser for the grammar-constrained single
# key/value YAML lane. Handles "output: value", optional surrounding quotes,
# and falls back to json.loads when the model ignored the grammar (defense:
# the GBNF grammar makes that impossible via ollama, but direct callers and
# llama.cpp paths bypass it). Returns the plain value string.
def _parse_yaml_output(text):
    if not text:
        return text
    stripped = text.strip()
    # JSON fallback: a model that ignored the YAML instruction
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "output" in parsed:
            return str(parsed["output"])
    except (json.JSONDecodeError, ValueError):
        pass
    # YAML single-key shape
    lines = stripped.splitlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith("output:"):
            val = line[len("output:"):].strip()
            val = val.strip('"').strip("'").strip()
            if val:
                return val
            # YAML block list ("output:" then "- item" lines), the shape
            # reasoning models pick for list answers (focal points): return
            # it as a JSON list string so list callers' literal_eval parses.
            # _strip_markdown has usually removed the "- " markers already.
            items = []
            for nxt in lines[idx + 1:]:
                nxt = nxt.strip()
                if nxt.startswith("- "):
                    nxt = nxt[2:].strip()
                nxt = nxt.strip('"').strip("'").strip()
                if nxt:
                    items.append(nxt)
            if items:
                return json.dumps(items)
    # free-form: return first non-empty line (grammar guarantees one)
    for line in stripped.splitlines():
        if line.strip():
            return line.strip()
    return stripped


def temp_sleep(seconds=0.1):
    time.sleep(seconds)


def _strip_markdown(text):
    """Strip markdown formatting and thinking blocks so parsers get clean plain text."""
    import re
    # Strip ```code fences``` (gemma3 wraps JSON outputs in these)
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text)
    # Strip <think>...</think> blocks (qwen3 thinking models leak these)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip </think> tag remnants without opening tag
    text = re.sub(r'</think>', '', text)
    # Truncated thinking (Oct 15 2026, LFM2.5): an OPEN tag with NO close
    # means generation died mid-reasoning at the num_predict cap —
    # everything from the OPEN onward is reasoning fragments, not an
    # answer. Cut back to the pre-OPEN text (usually empty) so the
    # caller's empty-content handling (think-native budget ladder in
    # _ollama_generate) sees it as empty instead of returning a bare
    # marker as if it were the answer.
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    # Remove bold/italic: **text** -> text, *text* -> text, __text__ -> text
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Remove headers: ## Title -> Title
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bullet list markers: "* item" or "- item" -> "item"
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    # Remove (Duration: X mins, Left: Y) parentheticals added by the model
    text = re.sub(r'\s*\(Duration:.*?\)', '', text)
    text = re.sub(r'\s*\(Left:.*?\)', '', text)
    # Remove Note: ... footnotes the model adds
    text = re.sub(r'\n\*?Note:.*', '', text, flags=re.DOTALL)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _llama_cpp_generate(prompt, retries=5, free_form=False):
    """
    Make a request to llama.cpp server's /completion endpoint.
    Returns the response text or raises an exception after retries exhausted.

    Uses GBNF grammar for JSON schema constraint (equivalent to Ollama's
    format field). The /completion endpoint returns {"content": "..."}
    instead of Ollama's {"response": "..."}.

    free_form=True: skip grammar constraint, return raw text.
    """
    url = f"{LLAMA_CPP_BASE_URL}/completion"

    system_msg = (
        "You are a simulation assistant. "
        "IMPORTANT: Respond in English only. "
    )
    if not free_form:
        system_msg += (
            "Output ONLY one line in the exact YAML format shown in the prompt: "
            "output: <value> "
            "No explanation, no markdown, no extra keys or lines."
        )
    else:
        system_msg += (
            "Follow the output format shown in the prompt EXACTLY. "
            "Use numbered lists like '1) task (duration in minutes: X, minutes left: Y)'. "
            "Do NOT use markdown tables, headers, or any formatting. "
            "Plain text numbered list only."
        )

    full_prompt = system_msg + "\n\n" + prompt

    request_body = {
        "prompt": full_prompt,
        "stream": False,
        "n_predict": 2048,
        "temperature": 0.7,
    }
    if not free_form:
        request_body["grammar"] = _GBNF_YAML_OUTPUT

    data = json.dumps(request_body).encode('utf-8')

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=None) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result.get("content", "")
                text = _strip_markdown(text)
                # Extract ["output"] from grammar-constrained JSON
                if text.strip():
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and "output" in parsed:
                            text = str(parsed["output"])
                    except (json.JSONDecodeError, ValueError):
                        pass
                    return text
                # Empty response — retry with backoff
                wait = 2 ** attempt
                print(f"[llama.cpp] Empty response on attempt {attempt+1}/{retries}, retrying in {wait}s...")
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"[llama.cpp] Error on attempt {attempt+1}/{retries}: {e}, retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
                continue
            raise

    print("[llama.cpp] WARNING: All retries exhausted, returning empty string")
    return ""


# Call capture (Oct 21 2026, n68): every successful /api/chat round-trip
# (prompt, system, answer, reasoning, tokens, latency, caller) appended to
# <repo>/ga_traces/calls/<model>.jsonl — distillation corpus for the sLM
# while the cloud teacher drives the sim. GA_CALL_TRACE=0 disables; fail-open:
# capture must never cost the sim a step.
_CALL_TRACE_LOCK = threading.Lock()
_CALL_TRACE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "ga_traces", "calls"))


def _call_site():
    """Nearest run_gpt_prompt_* / generate_* frame name (the call family)."""
    f = sys._getframe(2)
    while f is not None:
        n = f.f_code.co_name
        if n.startswith("run_gpt_prompt_") or n.startswith("generate_"):
            return n
        f = f.f_back
    return None


def _trace_call(request_body, result, latency_s, attempt):
    if os.environ.get("GA_CALL_TRACE", "1") != "1":
        return
    try:
        msg = result.get("message", {})
        rec = {
            "ts": time.time(),
            "persona": getattr(_PERSONA_CTX, "name", None),
            "caller": _call_site(),
            "model": request_body.get("model"),
            "think": request_body.get("think"),
            "options": request_body.get("options"),
            "system": request_body.get("system"),
            "prompt": request_body["messages"][0]["content"],
            "content": msg.get("content", ""),
            "thinking": msg.get("thinking", ""),
            "done_reason": result.get("done_reason"),
            "eval_count": result.get("eval_count"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "latency_s": round(latency_s, 3),
            "attempt": attempt,
        }
        name = str(rec["model"]).replace("/", "_").replace(":", "_")
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _CALL_TRACE_LOCK:
            os.makedirs(_CALL_TRACE_DIR, exist_ok=True)
            path = os.path.join(_CALL_TRACE_DIR, f"{name}.jsonl")
            # Size rotation (Oct 22 2026): the live file is committed with
            # each data snapshot — GitHub rejects files > 100 MB and every
            # commit stores a full copy. Roll over to <name>.partNNN.jsonl.
            try:
                rot = float(os.environ.get("GA_CALL_TRACE_ROT_MB", "20")) * 1024 * 1024
                if rot > 0 and os.path.getsize(path) >= rot:
                    n = 1
                    while os.path.exists(os.path.join(_CALL_TRACE_DIR, f"{name}.part{n:03d}.jsonl")):
                        n += 1
                    os.rename(path, os.path.join(_CALL_TRACE_DIR, f"{name}.part{n:03d}.jsonl"))
            except OSError:
                pass
            with open(path, "a") as fh:
                fh.write(line)
    except Exception:
        pass


def _ollama_generate(prompt, retries=5, free_form=False, model_hint=None):
    """
    Make a request to Ollama's CHAT endpoint (conversation-native; works with
    both classic completion models and modern reasoning models).

    2026-09 redesign: GA's prompt library was written for 2023-era completion
    APIs ("Plan:" style cues). Modern sLLMs are chat-native and often
    reasoning-first. Routing everything through /api/chat with explicit
    `think` control makes any model family usable:
      - think=false: no reasoning tokens, answer lands directly (fast lane)
      - think=true:  reasoning happens in a SEPARATE field, content stays
                     clean (heavy lanes can opt into quality)
    The prompt template text is sent verbatim as the user message — content
    compatibility is preserved; only the API surface changes.

    free_form=True: skip JSON format constraint (multi-line outputs like task
    decomposition). free_form=False: constrain to JSON object output.
    model_hint: call-family for multi-model routing (planning/artifact/convo/insight).
    """
    if USE_LLAMA_CPP:
        return _llama_cpp_generate(prompt, retries=retries, free_form=free_form)
    url = f"{OLLAMA_BASE_URL}/api/chat"
    model, num_ctx = _resolve_model(model_hint)
    # Think-native budgets (Oct 15 2026): modern reasoning models burn
    # 600-2500 reasoning tokens BEFORE the answer. A flat cap truncates
    # mid-reasoning (content arrives empty, done_reason=length) and the
    # old code answered that by suppressing thinking entirely. Adapt
    # instead: when the resolved call thinks, scale the answer budget by
    # GA_THINK_MULT (default 3x = 6144: reasoning headroom + the 2x-longest
    # legit answer shape). GA_THINK_MULT set explicitly applies even with
    # think off — always-think checkpoints (LFM2.5 family) ignore think=false
    # and need the headroom regardless of what we request.
    resolved_think = _persona_think(_lane_think(model_hint))
    _base_np = int(os.environ.get("GA_NUM_PREDICT", "2048"))
    _think_mult_env = os.environ.get("GA_THINK_MULT")
    if _think_mult_env is not None:
        num_predict = int(_base_np * float(_think_mult_env))
    elif resolved_think:
        num_predict = _base_np * 3
    else:
        num_predict = _base_np
    request_body = {
        "model": model,
        "stream": False,
        # Modern-model compatibility: suppress/enable reasoning explicitly.
        # Lane default is env-controlled (GA_<LANE>_THINK); per-persona
        # overrides come from town_model_settings.json (Layer 1).
        "think": resolved_think,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {
            "num_ctx": num_ctx,
            "temperature": _persona_temperature(0.7),
            # Runaway-generation cap (Oct 6 2026): the ollama redesign
            # dropped the legacy max_tokens cap, so a memo_on_convo call
            # generated 5,000+ tokens (one JSON sentence was the ask) while
            # the whole town waited behind CONVO_LOCK. The think-native
            # num_predict above keeps the cap; GA_NUM_PREDICT adjusts the
            # base (think lanes scale it via GA_THINK_MULT / resolved think).
            "num_predict": num_predict,
        }
    }
    if not free_form:
        # YAML lane (Oct 2026): "output: <value>" replaces format:"json".
        # NOTE: ollama's format field accepts ONLY "json" or a JSON schema —
        # a raw GBNF string 500s the request (verified live Oct 11: HTTP 500).
        # The YAML shape is enforced instruction-only here, which the A/B probe
        # showed is actually MORE reliable with qwen3.5-family models than
        # grammar-constrained JSON (12/12 vs 10/12 on the event triple).
        request_body["system"] = (
            "You are a simulation assistant. "
            "IMPORTANT: Respond in English only. "
            "Output ONLY one line in the exact YAML format shown in the prompt: "
            "output: <value> "
            "No explanation, no markdown, no extra keys or lines."
        )
    else:
        request_body["system"] = (
            "You are a simulation assistant. "
            "IMPORTANT: Respond in English only. "
            "Follow the output format shown in the prompt EXACTLY. "
            "Use numbered lists like '1) task (duration in minutes: X, minutes left: Y)'. "
            "Do NOT use markdown tables, headers, or any formatting. "
            "Plain text numbered list only."
        )
    data = json.dumps(request_body).encode('utf-8')

    # Think-native retry ladder (Oct 15 2026): a reasoning phase can outgrow
    # any fixed cap — content arrives empty with done_reason=length. The old
    # code suppressed thinking on the first miss (lazy think=false). Adapt
    # instead, in order: (1) ONE budget escalation (2x) when the response
    # died mid-reasoning at the cap — the reasoning may simply need more
    # room; (2) only on the NEXT empty response fall back to think=False —
    # last resort so a thinking checkpoint that ignores think=false still
    # can't freeze the town; (3) legacy backoff retries after that.
    # Trigger on done_reason=length regardless of the think flag: always-
    # think checkpoints (LFM2.5) reason even when told not to.
    _budget_escalated = False
    _forced_think = False
    for attempt in range(retries):
        try:
            if _forced_think and request_body.get("think"):
                request_body["think"] = False
                data = json.dumps(request_body).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            # local model, but bounded: an unbounded timeout let a wedged
            # ollama worker hang the whole backend silently (Sep 2026).
            # 300s: measured Oct 8 (econ1) — ornith-1.5:9b under 6-way load
            # generates ~5 t/s; a 750-token planning call brushes 180s, and
            # every timeout retry DISCARDS the partial generation and
            # re-queues it, feeding a saturation death-spiral (553 errors,
            # 3 steps/hour). Give big calls room to complete instead.
            # On timeout the retry loop gives the call 5 more chances.
            _t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
                _trace_call(request_body, result, time.time() - _t0, attempt)
                # /api/chat response: content lives under message.content.
                # If a model still emits reasoning inline (think=false
                # unsupported or ignored), the reasoning field is separate —
                # we only ever consume message.content, and _strip_markdown
                # removes the inline <think>-style block as a second line of
                # defense.
                msg = result.get("message", {})
                text = msg.get("content", "")
                if not text.strip() and result.get("response", ""):
                    text = result["response"]  # legacy/generate-shape fallback
                text = _strip_markdown(text)
                # YAML lane: unwrap "output: <value>" (grammar-constrained) or
                # fall back to JSON parsing if a caller bypassed the grammar.
                if text.strip():
                    # free_form answers are multi-line by contract (task
                    # decomp numbered lists): the single-line YAML unwrap
                    # used to cut them to their FIRST line (Oct 21 2026, n68:
                    # every decomp failed validation and burned 5 retries).
                    if not free_form:
                        text = _parse_yaml_output(text)
                    return text
                # Empty content. If the generation died AT the cap, the
                # reasoning phase swallowed the budget — adapt: double the
                # budget once before ever suppressing thinking.
                if (not _budget_escalated
                        and result.get("done_reason") == "length"):
                    _budget_escalated = True
                    request_body["options"]["num_predict"] = \
                        int(request_body["options"]["num_predict"]) * 2
                    data = json.dumps(request_body).encode('utf-8')
                    print(f"[Ollama] Reasoning outgrew the cap on attempt "
                          f"{attempt+1}/{retries}; retrying with 2x budget "
                          f"({request_body['options']['num_predict']} tokens)...")
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                # Still empty (or died for another reason) — fall back to
                # think=False so the answer lands in content, then retry
                # with backoff. A checkpoint that ignores the toggle will
                # simply run the ladder's escalated budget with thinking
                # anyway, which is the best it can do.
                _forced_think = True
                wait = 2 ** attempt
                print(f"[Ollama] Empty response on attempt {attempt+1}/{retries}, retrying in {wait}s...")
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError) as e:
            # TimeoutError: a mid-response read timeout (getresponse)
            # raises TimeoutError, NOT URLError — without it in the tuple
            # the timeout escaped the retry loop and killed the worker
            # thread (measured Oct 6: both A/B forks wedged on Eddy Lin's
            # revise_identity during the midnight burst).
            wait = 2 ** attempt
            print(f"[Ollama] Error on attempt {attempt+1}/{retries}: {e}, retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
                continue
            raise

    print("[Ollama] WARNING: All retries exhausted, returning empty string")
    return ""


def ChatGPT_single_request(prompt):
    """Single request to backend (llama.cpp or Ollama)."""
    temp_sleep()
    if USE_LLAMA_CPP:
        return _llama_cpp_generate(prompt)
    return _ollama_generate(prompt)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
    """
    Given a prompt, make a request to Ollama and return the response.
    (Replaces GPT-4 requests with local Ollama)

    ARGS:
      prompt: a str prompt
    RETURNS:
      a str of Ollama's response.
    """
    temp_sleep()
    try:
        if USE_LLAMA_CPP:
            return _llama_cpp_generate(prompt)
        return _ollama_generate(prompt)
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return "LLM ERROR"


def ChatGPT_request(prompt, model_hint=None):
    """
    Given a prompt, make a request to Ollama and return the response.
    (Replaces ChatGPT requests with local Ollama)

    ARGS:
      prompt: a str prompt
      model_hint: call-family for multi-model routing (planning/artifact/convo/insight)
    RETURNS:
      a str of Ollama's response.
    """
    try:
        if USE_LLAMA_CPP:
            return _llama_cpp_generate(prompt)
        return _ollama_generate(prompt, model_hint=model_hint)
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return "LLM ERROR"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in yaml. {special_instruction}\n"
    prompt += "Example output yaml:\n"
    prompt += 'output: "' + str(example_output) + '"'

    if verbose:
        print("OLLAMA PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass

    return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   model_hint=None):
    prompt = '"""' + '\n' + prompt + '\n' + '"""' + '\n'
    # YAML lane (Oct 2026): the in-prompt example switches from JSON to the
    # minimal YAML shape; transport grammar enforces it at the sampler.
    prompt += f"Output the response to the prompt above in yaml. {special_instruction}\n"
    prompt += "Example output yaml:\n"
    prompt += 'output: "' + str(example_output) + '"'

    # FIX (vector 2 — deterministic call cache): many GA prompts are fully
    # deterministic (temp 0) and repeat identically every step — sleeping
    # agents re-ask "what emoji for sleeping?" hundreds of times. Cache the
    # FINAL cleaned result keyed on the exact prompt + model.
    global _RESPONSE_CACHE
    try:
        _RESPONSE_CACHE
    except NameError:
        _RESPONSE_CACHE = {}
    _cache_key = None
    try:
        import hashlib as _hashlib
        _cache_key = (_hashlib.md5(prompt.encode()).hexdigest(),
                      str(model_hint or OLLAMA_CHAT_MODEL))
        if _cache_key in _RESPONSE_CACHE:
            return _RESPONSE_CACHE[_cache_key]
    except Exception:
        _cache_key = None

    if verbose:
        print("OLLAMA PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            # _ollama_generate already extracts ["output"] from schema-constrained JSON,
            # so curr_gpt_response is already the plain value (e.g. "5", "Maria is sleeping").
            curr_gpt_response = ChatGPT_request(prompt, model_hint=model_hint).strip()

            if func_validate(curr_gpt_response, prompt=prompt):
                result = func_clean_up(curr_gpt_response, prompt=prompt)
                if _cache_key is not None and len(str(result)) < 512:
                    _RESPONSE_CACHE[_cache_key] = result
                return result

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass

    # Return fail_safe instead of False so callers that do result[0] don't crash.
    return fail_safe_response


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
    if verbose:
        print("OLLAMA PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter, free_form=False, model_hint=None):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to Ollama.
    gpt_parameter is accepted for API compatibility but ignored — Ollama uses
    schema-constrained JSON format instead. Cleanup functions handle output parsing.

    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary (accepted for compat, not forwarded)
      free_form: if True, skip schema constraint (for multi-line outputs like task decomp)
      model_hint: call-family for multi-model routing (planning/artifact/convo/insight)
    RETURNS:
      a str of Ollama's response.
    """
    temp_sleep()
    try:
        if USE_LLAMA_CPP:
            return _llama_cpp_generate(prompt, free_form=free_form)
        return _ollama_generate(prompt, free_form=free_form, model_hint=model_hint)
    except Exception as e:
        print(f"TOKEN LIMIT EXCEEDED: {e}")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classify) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final prompt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the prompt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    prompt = prompt.strip()
    # Force English output — prepend instruction to every prompt
    prompt = "IMPORTANT: Respond in English only.\n" + prompt
    return prompt


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           free_form=False,
                           model_hint=None,
                           cache=True):
    if verbose:
        print(prompt)

    # PERF (improvement 1 — deterministic prompt cache): the n30 telemetry showed
    # 99% of (agent, description) pairs repeat across steps, yet event_triple /
    # sector / arena / poig lanes (safe_generate_response) had NO cache — only
    # ChatGPT_safe_generate_response did. Cache the FINAL validated result keyed
    # on (md5(prompt), model_hint). Novelty-dependent callers (conversation
    # generation, next-line, whisper) must pass cache=False.
    global _SAFE_RESPONSE_CACHE
    try:
        _SAFE_RESPONSE_CACHE
    except NameError:
        _SAFE_RESPONSE_CACHE = {}
    _safe_cache_key = None
    try:
        import hashlib as _hashlib
        _safe_cache_key = (_hashlib.md5(prompt.encode()).hexdigest(),
                           str(model_hint or OLLAMA_CHAT_MODEL))
        if cache and _safe_cache_key in _SAFE_RESPONSE_CACHE:
            return _SAFE_RESPONSE_CACHE[_safe_cache_key]
    except Exception:
        _safe_cache_key = None

    for i in range(repeat):
        try:
            curr_gpt_response = GPT_request(prompt, gpt_parameter, free_form=free_form,
                                            model_hint=model_hint)
            if func_validate(curr_gpt_response, prompt=prompt):
                result = func_clean_up(curr_gpt_response, prompt=prompt)
                if cache and _safe_cache_key is not None and len(str(result)) < 2048:
                    _SAFE_RESPONSE_CACHE[_safe_cache_key] = result
                return result
            if verbose:
                print("---- repeat count: ", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            # A transport/validation exception must NEVER escape to reverie's
            # interactive console (that wedges the backend silently — the
            # 2026-09-23 bulletin-board incident). Swallow, count, retry.
            print(f"[safe_generate_response] attempt {i+1}/{repeat} exception: {e!r}")
    return fail_safe_response


def _find_llama_cpp_pid():
    """Find the PID of the running llama-server process."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server.*--port"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        return int(pids[0]) if pids and pids[0] else None
    except (ValueError, IndexError):
        return None


def _is_llama_cpp_running():
    """Check if llama.cpp server is reachable."""
    try:
        req = urllib.request.Request(f"{LLAMA_CPP_BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _stop_llama_cpp():
    """Stop llama.cpp server and wait for VRAM to free."""
    import signal, subprocess
    pid = _find_llama_cpp_pid()
    if pid:
        print(f"[embedding] Stopping llama.cpp (PID {pid}) to free VRAM...")
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit and VRAM to free
            for _ in range(30):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)  # check if still alive
                except OSError:
                    break
            time.sleep(2)  # extra wait for VRAM release
            return True
        except OSError:
            pass
    return False


def _start_llama_cpp():
    """Restart llama.cpp server in background."""
    import subprocess
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "llama_cpp_server", "start_server.sh"
    )
    if not os.path.exists(script):
        print("[embedding] Warning: start_server.sh not found, cannot restart llama.cpp")
        return False

    # Find the model path from /proc before we killed it, or use nohup.out
    # Use the GGUF path from environment or a known default
    model_path = os.environ.get("LLAMA_GGUF_PATH", "")
    if not model_path:
        # Try to find from nohup.out or recent process args
        try:
            nohup = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))))),
                "nohup.out"
            )
            if os.path.exists(nohup):
                with open(nohup) as f:
                    for line in f:
                        if "Model:" in line and ".gguf" in line:
                            model_path = line.split("Model:")[-1].strip()
                            break
        except Exception:
            pass

    if not model_path:
        print("[embedding] Warning: cannot determine GGUF model path, set LLAMA_GGUF_PATH env var")
        return False

    print(f"[embedding] Restarting llama.cpp server with model: {model_path}")
    subprocess.Popen(
        ["bash", script, model_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    # Wait for server to become healthy
    for _ in range(60):
        time.sleep(1)
        if _is_llama_cpp_running():
            print("[embedding] llama.cpp server is back up")
            return True
    print("[embedding] Warning: llama.cpp server did not come back up within 60s")
    return False


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get embedding vector for text.
    If USE_LLAMA_CPP=True: uses llama.cpp /embedding endpoint (no Ollama needed).
    Otherwise: uses Ollama embeddinggemma.
    """
    # FIX (reflection wedge, n33 step 7962): a list-of-dicts focal point from
    # the LLM reached here and AttributeError'd the whole backend into its
    # interactive console. Never let a type error kill a 17k-step run.
    if not isinstance(text, str):
        if isinstance(text, dict):
            text = " ".join(str(v) for v in text.values())
        else:
            text = str(text)
        print(f"[embedding] Warning: non-str input coerced: {text[:80]!r}")
    text = text.replace("\n", " ").strip()
    if not text:
        text = "this is blank"

    # FIX (vector 2 — embedding cache): identical strings get embedded over
    # and over (~110ms each). A simple dict makes repeats free and is bounded
    # by the number of distinct short strings GA produces.
    global _EMBED_CACHE
    try:
        _EMBED_CACHE
    except NameError:
        _EMBED_CACHE = {}
    cache_key = text[:512]  # long texts truncated consistently
    if cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[cache_key]
    _orig_text = text

    if USE_LLAMA_CPP:
        # Use llama.cpp /embedding endpoint directly — no Ollama required
        url = f"{LLAMA_CPP_BASE_URL}/embedding"
        data = json.dumps({"content": text}).encode("utf-8")
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    # Response: [{\"index\": 0, \"embedding\": [[...2688 floats...]]} ]
                    if isinstance(result, list) and result:
                        emb = result[0].get("embedding", [])
                        # emb is [[floats]] — take the first token vector
                        if emb and isinstance(emb[0], list):
                            emb = emb[0]
                        if emb:
                            _EMBED_CACHE[cache_key] = emb
                        return emb
            except Exception as e:
                print(f"[embedding] llama.cpp error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(1)
        return []

    # Ollama fallback
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    data = json.dumps({
        "model": OLLAMA_EMBED_MODEL,
        "prompt": _orig_text
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=None) as response:
                result = json.loads(response.read().decode("utf-8"))
                embedding = result.get("embedding", [])
                if embedding:
                    _EMBED_CACHE[cache_key] = embedding
                    return embedding
                if attempt < 2:
                    time.sleep(1)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
    return []

if __name__ == '__main__':
    gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response, prompt=None):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True

    def __func_clean_up(gpt_response, prompt=None):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    output = safe_generate_response(prompt,
                                    gpt_parameter,
                                    5,
                                    "rest",
                                    __func_validate,
                                    __func_clean_up,
                                    True)

    print(output)
