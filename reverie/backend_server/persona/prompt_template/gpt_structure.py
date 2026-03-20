"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified: Local Ollama adaptation

File: gpt_structure.py
Description: Wrapper functions for calling local Ollama APIs instead of OpenAI.
"""
import json
import time
import urllib.request
import urllib.error

from utils import *

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "claude-distilled-v2:9b"
OLLAMA_EMBED_MODEL = "embeddinggemma"


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


def _ollama_generate(prompt, retries=5, free_form=False):
    """
    Make a request to Ollama's generate endpoint.
    Returns the response text or raises an exception after retries exhausted.
    Retries with exponential backoff on empty responses.
    Schema-constrained JSON ensures output is always {"output": "..."}.
    Individual func_clean_up functions handle parsing the output value.

    free_form=True: skip schema constraint, return raw text (for multi-line prompts
    like task decomposition that need numbered list output, not a JSON string).
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    request_body = {
        "model": OLLAMA_CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.7,
        }
    }
    if not free_form:
        request_body["format"] = {"type": "object", "properties": {"output": {"type": "string"}}, "required": ["output"]}  # schema-constrained: always produces {"output": "..."}
        request_body["system"] = (
            "You are a simulation assistant. "
            "IMPORTANT: Respond in English only. "
            "Output valid JSON exactly matching the example format in the prompt. "
            "Do not add any explanation, markdown, or extra keys."
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

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=None) as response:  # no timeout - local model, let it run
                result = json.loads(response.read().decode('utf-8'))
                text = result.get("response", "")
                # qwen3.5 is a thinking model — response is in response field when complete,
                # but if empty, extract from thinking field as fallback
                if not text.strip() and result.get("thinking"):
                    thinking = result["thinking"]
                    # Use the last coherent sentence/answer from thinking
                    text = thinking.strip()
                text = _strip_markdown(text)
                # Extract ["output"] value from schema-constrained JSON.
                # The format enforcement always wraps responses as {"output": "..."},
                # but most func_clean_up callers expect plain text, not JSON.
                if text.strip():
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and "output" in parsed:
                            text = str(parsed["output"])
                    except (json.JSONDecodeError, ValueError):
                        pass  # not JSON — use as-is
                    return text
                # Empty response — wait and retry with backoff
                wait = 2 ** attempt
                print(f"[Ollama] Empty response on attempt {attempt+1}/{retries}, retrying in {wait}s...")
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"[Ollama] Error on attempt {attempt+1}/{retries}: {e}, retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
                continue
            raise

    print("[Ollama] WARNING: All retries exhausted, returning empty string")
    return ""


def ChatGPT_single_request(prompt):
    """Single request to Ollama (replaces ChatGPT single request)."""
    temp_sleep()
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
        return _ollama_generate(prompt)
    except Exception as e:
        print(f"Ollama ERROR: {e}")
        return "Ollama ERROR"


def ChatGPT_request(prompt):
    """
    Given a prompt, make a request to Ollama and return the response.
    (Replaces ChatGPT requests with local Ollama)

    ARGS:
      prompt: a str prompt
    RETURNS:
      a str of Ollama's response.
    """
    try:
        return _ollama_generate(prompt)
    except Exception as e:
        print(f"Ollama ERROR: {e}")
        return "Ollama ERROR"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

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
                                   verbose=False):
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("OLLAMA PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            # _ollama_generate already extracts ["output"] from schema-constrained JSON,
            # so curr_gpt_response is already the plain value (e.g. "5", "Maria is sleeping").
            curr_gpt_response = ChatGPT_request(prompt).strip()

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

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

def GPT_request(prompt, gpt_parameter, free_form=False):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to Ollama.
    gpt_parameter is accepted for API compatibility but ignored — Ollama uses
    schema-constrained JSON format instead. Cleanup functions handle output parsing.

    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary (accepted for compat, not forwarded)
      free_form: if True, skip schema constraint (for multi-line outputs like task decomp)
    RETURNS:
      a str of Ollama's response.
    """
    temp_sleep()
    try:
        return _ollama_generate(prompt, free_form=free_form)
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
                           free_form=False):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter, free_form=free_form)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get embedding vector for text using Ollama's nomic-embed-text model.

    ARGS:
      text: the text to embed
      model: ignored (we always use nomic-embed-text)
    RETURNS:
      a list of floats representing the embedding vector
    """
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    data = json.dumps({
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
    }).encode('utf-8')

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                embedding = result.get("embedding", [])
                if embedding:
                    return embedding
                # Empty response, retry
                if attempt < 2:
                    time.sleep(1)
                    continue
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

    # Return empty list if all retries fail
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
