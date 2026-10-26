"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py
"""
import os
import re
import string
import random
import json
from os import listdir
import os

import datetime
import time
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse, JsonResponse
from global_methods import *

from django.templatetags.static import static
from .models import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def landing(request): 
  context = {}
  template = "landing/landing.html"
  return render(request, template, context)


def demo(request, sim_code, step, play_speed="2"): 
  move_file = f"compressed_storage/{sim_code}/master_movement.json"
  meta_file = f"compressed_storage/{sim_code}/meta.json"
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt: play_speed = 2
  else: play_speed = play_speed_opt[play_speed]

  # Loading the basic meta information about the simulation.
  meta = dict() 
  with open (meta_file) as json_file: 
    meta = json.load(json_file)

  sec_per_step = meta["sec_per_step"]
  start_datetime = datetime.datetime.strptime(meta["start_date"] + " 00:00:00", 
                                              '%B %d, %Y %H:%M:%S')
  for i in range(step): 
    start_datetime += datetime.timedelta(seconds=sec_per_step)
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  # Loading the movement file
  raw_all_movement = dict()
  with open(move_file) as json_file: 
    raw_all_movement = json.load(json_file)
 
  # Loading all names of the personas
  persona_names = dict()
  persona_names = []
  persona_names_set = set()
  for p in list(raw_all_movement["0"].keys()): 
    persona_names += [{"original": p, 
                       "underscore": p.replace(" ", "_"), 
                       "initial": p[0] + p.split(" ")[-1][0]}]
    persona_names_set.add(p)

  # <all_movement> is the main movement variable that we are passing to the 
  # frontend. Whereas we use ajax scheme to communicate steps to the frontend
  # during the simulation stage, for this demo, we send all movement 
  # information in one step. 
  all_movement = dict()

  # Preparing the initial step. 
  # <init_prep> sets the locations and descriptions of all agents at the
  # beginning of the demo determined by <step>. 
  init_prep = dict() 
  for int_key in range(step+1): 
    key = str(int_key)
    val = raw_all_movement[key]
    for p in persona_names_set: 
      if p in val: 
        init_prep[p] = val[p]
  persona_init_pos = dict()
  for p in persona_names_set: 
    persona_init_pos[p.replace(" ","_")] = init_prep[p]["movement"]
  all_movement[step] = init_prep

  # Finish loading <all_movement>
  for int_key in range(step+1, len(raw_all_movement.keys())): 
    all_movement[int_key] = raw_all_movement[str(int_key)]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos), 
             "all_movement": json.dumps(all_movement), 
             "start_datetime": start_datetime,
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  template = "demo/demo.html"

  return render(request, template, context)


def UIST_Demo(request): 
  return demo(request, "March20_the_ville_n25_UIST_RUN-step-1-141", 2160, play_speed="3")


def home(request): 
  f_curr_sim_code = "temp_storage/curr_sim_code.json"
  f_curr_step = "temp_storage/curr_step.json"

  if not check_if_file_exists(f_curr_step): 
    context = {}
    template = "home/error_start_backend.html"
    return render(request, template, context)

  with open(f_curr_sim_code) as json_file:  
    sim_code = json.load(json_file)["sim_code"]
  
  with open(f_curr_step) as json_file: 
    step = json.load(json_file)["step"]

  # NOTE: We intentionally do NOT delete curr_step.json here.
  # The original Stanford code deleted it to enforce one-shot initialization,
  # but that breaks page refreshes. Leaving it allows re-initialization from
  # the latest known step on refresh. The backend (reverie.py) continuously
  # overwrites it, so it always reflects the current step.

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".":
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  # Load initial positions from the environment file matching the starting
  # step, not the latest file. Using max(file_count) would load positions
  # from a much later step after a fork, causing sprites to start at wrong
  # locations and desync with the backend.
  # If the exact step file is missing (backend's curr_step runs ahead of
  # the environment files under concurrency — browser POSTs race the
  # headless driver), fall back to the newest existing file instead of a
  # 500 FileNotFoundError.
  curr_json = f'storage/{sim_code}/environment/{str(step)}.json'
  if not check_if_file_exists(curr_json):
    env_dir = f'storage/{sim_code}/environment'
    candidates = sorted(
      (int(f.split(".")[0]) for f in os.listdir(env_dir)
       if f.endswith(".json") and f.split(".")[0].isdigit()),
      reverse=True)
    if candidates:
      step = candidates[0]
      curr_json = f'storage/{sim_code}/environment/{str(step)}.json'
  with open(curr_json) as json_file: 
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items():
      if key in persona_names_set:
        persona_init_pos += [[key, val["x"], val["y"]]]

  # Read-only follow mode is the DEFAULT (Oct 21 2026): every run is driven
  # by headless_driver.py, the sole writer of environment files — a browser
  # in "simulate" mode POSTs positions too and races it. ?drive=1 restores
  # the legacy browser-driven mode (no headless driver running).
  mode = "simulate" if request.GET.get("drive") else "replay"

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos,
             "mode": mode}
  template = "home/home.html"
  return render(request, template, context)


def replay(request, sim_code, step): 
  sim_code = sim_code
  step = int(step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  curr_json = f'storage/{sim_code}/environment/{str(step)}.json'
  with open(curr_json) as json_file:
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items():
      if key in persona_names_set:
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos,
             "mode": "replay"}
  template = "home/home.html"
  return render(request, template, context)


def replay_persona_state(request, sim_code, step, persona_name): 
  sim_code = sim_code
  step = int(step)

  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  memory = f"storage/{sim_code}/personas/{persona_name}/bootstrap_memory"
  if not os.path.exists(memory): 
    memory = f"compressed_storage/{sim_code}/personas/{persona_name}/bootstrap_memory"

  with open(memory + "/scratch.json") as json_file:  
    scratch = json.load(json_file)

  with open(memory + "/spatial_memory.json") as json_file:  
    spatial = json.load(json_file)

  with open(memory + "/associative_memory/nodes.json") as json_file:  
    associative = json.load(json_file)

  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  for count in range(len(associative.keys()), 0, -1): 
    node_id = f"node_{str(count)}"
    node_details = associative[node_id]

    if node_details["type"] == "event":
      a_mem_event += [node_details]

    elif node_details["type"] == "chat":
      a_mem_chat += [node_details]

    elif node_details["type"] == "thought":
      a_mem_thought += [node_details]
  
  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name, 
             "persona_name_underscore": persona_name_underscore, 
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  template = "persona_state/persona_state.html"
  return render(request, template, context)


def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html"
  return render(request, template, context)


def process_environment(request): 
  """
  <FRONTEND to BACKEND> 
  This sends the frontend visual world information to the backend server. 
  It does this by writing the current environment representation to 
  "storage/environment.json" file. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]
  environment = data["environment"]

  with open(f"storage/{sim_code}/environment/{step}.json", "w") as outfile:
    outfile.write(json.dumps(environment, indent=2))

  return HttpResponse("received")


def update_environment(request): 
  """
  <BACKEND to FRONTEND> 
  This sends the backend computation of the persona behavior to the frontend
  visual server. 
  It does this by reading the new movement information from 
  "storage/movement.json" file.

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]

  response_data = {"<step>": -1}
  if (check_if_file_exists(f"storage/{sim_code}/movement/{step}.json")):
    with open(f"storage/{sim_code}/movement/{step}.json") as json_file: 
      response_data = json.load(json_file)
      response_data["<step>"] = step

  # Viewer resync (Oct 21 2026): the page's step counter is fixed at load,
  # so after a checkpoint rollback / restart it waited forever for a step
  # being re-generated, and after a fork it kept polling the dead sim —
  # the run looked stalled. Report the live frontier + sim so the page can
  # re-initialise itself.
  response_data["frontier"] = _movement_frontier(sim_code)
  try:
    with open("temp_storage/curr_sim_code.json") as f:
      response_data["live_sim_code"] = json.load(f)["sim_code"]
  except (OSError, ValueError, KeyError):
    response_data["live_sim_code"] = None

  return JsonResponse(response_data)


def _movement_frontier(sim_code):
  """Newest movement step for a sim, or None."""
  try:
    steps = [int(f[:-5]) for f in os.listdir(f"storage/{sim_code}/movement")
             if f.endswith(".json") and f[:-5].isdigit()]
    return max(steps) if steps else None
  except OSError:
    return None


def path_tester_update(request):
  """
  Processing the path and saving it to path_tester_env.json temp storage for
  conducting the path tester.

  ARGS:
    request: Django request
  RETURNS:
    HttpResponse: string confirmation message.
  """
  data = json.loads(request.body)
  camera = data["camera"]

  with open(f"temp_storage/path_tester_env.json", "w") as outfile:
    outfile.write(json.dumps(camera, indent=2))

  return HttpResponse("received")


def get_agent_needs(request, sim_code, persona_name):
    """
    API endpoint to get agent needs for a persona.
    Returns JSON with needs levels (0-100) for hunger, thirst, energy, hygiene, etc.

    Works for both live simulations (storage/) and compressed demos (compressed_storage/).
    """
    persona_name_spaced = persona_name.replace("_", " ")

    # Try live storage first, then compressed
    scratch_paths = [
        f"storage/{sim_code}/personas/{persona_name_spaced}/bootstrap_memory/scratch.json",
        f"compressed_storage/{sim_code}/personas/{persona_name_spaced}/bootstrap_memory/scratch.json",
    ]

    scratch_data = None
    for path in scratch_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    scratch_data = json.load(f)
                break
            except (json.JSONDecodeError, IOError):
                continue

    if not scratch_data:
        return JsonResponse({"error": "Persona not found", "persona": persona_name_spaced}, status=404)

    # Extract needs data with defaults for backwards compatibility
    needs = scratch_data.get("needs", {
        "hunger": 80,
        "energy": 80,
        "hydration": 80,
        "hygiene": 80,
        "bladder": 80,
        "social": 80,
        "comfort": 80,
        "stimulation": 80,
    })

    # Include resource goals if present
    resource_goals = scratch_data.get("resource_goals", [])

    return JsonResponse({
        "persona": persona_name_spaced,
        "needs": needs,
        "resource_goals": resource_goals,
        "needs_critical": scratch_data.get("needs_critical", 20),
        "needs_danger": scratch_data.get("needs_danger", 5),
    })


def get_agent_cards(request, sim_code):
    """Live wallet / needs / fridge / job per agent (backend snapshot every ~1 sim-minute)."""
    for base in ("storage", "compressed_storage"):
        path = f"{base}/{sim_code}/resources/agent_cards.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return JsonResponse(json.load(f))
            except (json.JSONDecodeError, IOError):
                break
    return JsonResponse({})


def get_world_resources(request, sim_code):
    resource_paths = [
        f"storage/{sim_code}/resources/world_state.json",
        f"compressed_storage/{sim_code}/resources/world_state.json",
    ]

    world_state = None
    for path in resource_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    world_state = json.load(f)
                break
            except (json.JSONDecodeError, IOError):
                continue

    if not world_state:
        return JsonResponse({"resources": {}, "message": "No resource state found"})

    resources = {}
    for address, items in world_state.items():
        processed = {}
        for item, value in items.items():
            if item in ("max", "last_restocked", "last_delivery", "refill_rate"):
                continue
            if isinstance(value, (int, float)):
                max_val = items.get("max", 100)
                if item == "level":
                    processed["level_pct"] = int((value / max_val) * 100) if max_val > 0 else 0
                    processed["level"] = value
                else:
                    processed[item] = value

        if processed:
            resources[address] = processed

    return JsonResponse({"resources": resources})


def get_artifacts(request, sim_code):
    artifact_paths = [
        os.path.join(PROJECT_ROOT, f"environment/frontend_server/storage/{sim_code}/artifacts/artifacts.json"),
        os.path.join(PROJECT_ROOT, f"environment/frontend_server/compressed_storage/{sim_code}/artifacts/artifacts.json"),
    ]

    artifacts_data = None
    for path in artifact_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    artifacts_data = json.load(f)
                break
            except (json.JSONDecodeError, IOError):
                continue

    if not artifacts_data:
        return JsonResponse({"artifacts": [], "message": "No artifacts found"})

    artifacts_list = []
    for artifact_id, artifact in artifacts_data.items():
        artifacts_list.append({
            "id": artifact_id,
            "type": artifact.get("type", "unknown"),
            "name": artifact.get("name", "Unnamed"),
            "description": artifact.get("description", ""),
            "content_summary": artifact.get("content_summary", ""),
            "creator": artifact.get("creator", "Unknown"),
            "created_time": artifact.get("created_time", ""),
            "quality": artifact.get("quality", 0),
            "location": artifact.get("location", ""),
            "content_full": artifact.get("content_full", ""),
        })

    artifacts_list.sort(key=lambda x: x.get("created_time", ""), reverse=True)
    return JsonResponse({"artifacts": artifacts_list})


def get_artifact_content(request, sim_code, artifact_type, filename):
    if artifact_type not in ("writings", "paintings", "songs", "meals", "inventions", "images"):
        return JsonResponse({"error": "Invalid artifact type"}, status=400)

    content_paths = [
        os.path.join(PROJECT_ROOT, f"reverie/backend_server/artifacts/{artifact_type}/{filename}"),
        os.path.join(PROJECT_ROOT, f"storage/{sim_code}/artifacts/{artifact_type}/{filename}"),
        os.path.join(PROJECT_ROOT, f"compressed_storage/{sim_code}/artifacts/{artifact_type}/{filename}"),
    ]

    for path in content_paths:
        if os.path.exists(path):
            try:
                if artifact_type == "images" or filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
                    with open(path, "rb") as f:
                        ctype = "image/png" if filename.endswith(".png") else "image/jpeg"
                        return HttpResponse(f.read(), content_type=ctype)
                else:
                    with open(path) as f:
                        content = f.read()
                    return JsonResponse({"content": content, "filename": filename, "type": artifact_type})
            except (IOError, OSError):
                continue

    return JsonResponse({"error": "Artifact content not found"}, status=404)


def artifact_viewer(request, sim_code):
    template = "artifacts/artifacts.html"
    context = {"sim_code": sim_code}
    return render(request, template, context)


def _backend_proc_env_gates():
    """Read the GA_* decision-engine gates from the live reverie backend's
    /proc environ. Returns {} if no backend is running."""
    import subprocess
    try:
        pids = subprocess.run(["pgrep", "-f", "reverie.py"],
                              capture_output=True, text=True, timeout=5)
        out = {}
        for pid in pids.stdout.split():
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    for ent in f.read().decode("utf-8", "replace").split("\0"):
                        if ent.startswith("GA_"):
                            k, _, v = ent.partition("=")
                            out[k] = v
            except (OSError, PermissionError):
                continue
        return out
    except Exception:
        return {}


def _backend_unit():
    """systemd unit name owning the live reverie backend, or None."""
    import subprocess
    try:
        pids = subprocess.run(["pgrep", "-f", "reverie.py"],
                              capture_output=True, text=True, timeout=5)
        for pid in pids.stdout.split():
            try:
                with open(f"/proc/{pid}/cgroup") as f:
                    units = [seg for seg in f.read().strip().split("/")
                             if seg.endswith(".service")]
            except OSError:
                continue
            if units:
                return units[-1]
    except Exception:
        pass
    return None


def _probe_service(port, path="/health"):
    """Tiny HTTP probe of a local decision service; returns 'up'/'down'.
    Laya's server only implements do_POST, so POST a minimal typed decision
    when a bare GET would 501."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
        return "up"
    except urllib.error.HTTPError as e:
        # a response with ANY status means the service is listening
        if 400 <= e.code < 600 and e.code not in (501, 502, 503, 504):
            return "up"
        if port == 8018:
            # laya speaks POST-only /v1/systemone; 501/502/503 on GET are
            # just the stdlib server refusing unsupported verbs
            body = json.dumps({"model": "probe", "state": {"who": "probe", "event": "health check"},
                              "questions": {"poignancy": {"type": "score", "instructions": "x",
                                                          "criteria": ["1 = a", "10 = b"]}}}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=body,
                                          headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=3)
                return "up"
            except Exception:
                return "down"
        return "down"
    except Exception:
        return "down"


def _step_activity(frontier_file):
    """{step_age_s, calls_since_step, last_call} from the frontier file's
    mtime and the tail of the newest ga_traces/calls/*.jsonl; None on error."""
    import glob
    try:
        calls_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "ga_traces", "calls"))
        files = glob.glob(os.path.join(calls_dir, "*.jsonl"))
        now = time.time()
        step_mtime = os.path.getmtime(frontier_file) if frontier_file else None
        out = {"step_age_s": round(now - step_mtime) if step_mtime else None,
               "calls_since_step": None, "last_call": None}
        if not files:
            return out
        newest = max(files, key=os.path.getmtime)
        with open(newest, "rb") as f:
            f.seek(max(0, os.path.getsize(newest) - 256_000))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
        recs = []
        for line in lines:
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
        if not recs:
            return out
        if step_mtime:
            out["calls_since_step"] = sum(1 for r in recs if r.get("ts", 0) > step_mtime)
        last = recs[-1]
        out["last_call"] = {
            "caller": (last.get("caller") or "").replace("run_gpt_prompt_", ""),
            "persona": last.get("persona"),
            "age_s": round(now - last.get("ts", now)),  # ts = completion time
            "latency_s": last.get("latency_s"),
        }
        return out
    except Exception:
        return None


def get_run_stats(request, sim_code):
    """Live run stats for the simulator_home ops panel.

    Sources (all cheap, no heavy loads):
    - reverie/meta.json: checkpoint step + sim clock
    - movement dir: latest step + step rate from the newest file mtimes
    - /proc of the reverie backend: decision-engine gates in the live process
    - decision services: laya :8018 / rizzo :8017 / ga-ollama :11436 probes
    - nvidia-smi: GPU util + VRAM (if present)
    - ga-sim.service: chain unit state via systemctl --user
    """
    import subprocess
    import statistics

    stats = {"sim_code": sim_code}

    # --- checkpoint ---
    try:
        with open(f"storage/{sim_code}/reverie/meta.json") as f:
            meta = json.load(f)
        stats["checkpoint_step"] = meta.get("step")
        stats["sim_time"] = meta.get("curr_time")
    except (IOError, ValueError):
        stats["checkpoint_step"] = None
        stats["sim_time"] = None

    # --- movement frontier + step rate (trailing 10-min window) ---
    # Rate = steps written in the last 10 min / time since the oldest of
    # them. The old "newest 60 mtimes" span read 0.0 after every resume:
    # the files at/below the resume point are inherited from the parent
    # run (hours-old mtimes), so the span exploded. A frozen sim now
    # correctly decays to 0.0 instead of reporting its last burst.
    mdir = f"storage/{sim_code}/movement"
    latest_step, rate = None, None
    RATE_WINDOW_S = 600
    try:
        names = sorted((int(f[:-5]) for f in os.listdir(mdir)
                        if f.endswith(".json") and f[:-5].isdigit()))
        if names:
            latest_step = names[-1]
            now = time.time()
            mtimes = [os.path.getmtime(os.path.join(mdir, f"{n}.json"))
                      for n in names[-300:]]
            recent = [m for m in mtimes if now - m <= RATE_WINDOW_S]
            if recent:
                span = max(now - min(recent), 60)
                rate = round(len(recent) / span * 60, 1)
            else:
                rate = 0.0
    except OSError:
        pass
    stats["latest_step"] = latest_step
    stats["steps_per_min"] = rate
    # Live sim clock from the frontier file (meta.json only updates at
    # checkpoints, so it lags the running sim by hours).
    if latest_step is not None:
        try:
            with open(os.path.join(mdir, f"{latest_step}.json")) as f:
                stats["sim_time"] = json.load(f)["meta"]["curr_time"]
        except (OSError, ValueError, KeyError):
            pass

    # --- in-step activity (Oct 21 2026) ---
    # One step can take minutes (a whole conversation + re-plan is generated
    # inside a single step), which used to read as a stall. The per-call
    # capture (ga_traces/calls/<model>.jsonl, gpt_structure._trace_call)
    # shows whether the backend is actually working inside the step.
    stats["activity"] = _step_activity(
        os.path.join(mdir, f"{latest_step}.json") if latest_step is not None else None)

    # --- backend gates (live process env) ---
    gates = _backend_proc_env_gates()
    stats["engine"] = {
        "laya_enabled": gates.get("GA_LAYA_POIGNANCY") == "1",
        "rizzo_fallback": gates.get("GA_RIZZO_POIGNANCY") == "1",
        "rizzo_reflex": gates.get("GA_RIZZO_REFLEX") == "1",
        "cloudjev": gates.get("GA_CLOUDJEV") == "1",
        "backend_pid": bool(gates),
        "chat_model": gates.get("GA_CHAT_MODEL", ""),
        "artifact_model": gates.get("GA_ARTIFACT_MODEL", ""),
    }

    # --- decision services ---
    stats["services"] = {
        "laya": _probe_service(8018, "/v1/systemone"),
        "rizzo": _probe_service(8017, "/health"),
        "ga_ollama": _probe_service(11436, "/api/version"),
    }

    # --- chain unit ---
    # The driving unit changes per run (ga-sim, ga-n68, ...): resolve it from
    # the live backend's cgroup; fall back to the legacy ga-sim.service.
    unit = _backend_unit() or "ga-sim.service"
    stats["chain_unit_name"] = unit
    try:
        u = subprocess.run(["systemctl", "--user", "is-active", unit],
                           capture_output=True, text=True, timeout=5)
        stats["chain_unit"] = u.stdout.strip()
    except Exception:
        stats["chain_unit"] = "unknown"

    # --- GPU ---
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        parts = g.stdout.strip().split(", ")
        if len(parts) == 3:
            stats["gpu"] = {"util": int(parts[0]), "vram_used": int(parts[1]),
                            "vram_total": int(parts[2])}
    except Exception:
        stats["gpu"] = None

    # --- backend log: [LAYA] errors in the last ~2000 lines ---
    stats["laya_errors"] = None
    try:
        with open(f"/tmp/{sim_code}_backend.log", "rb") as f:
            f.seek(max(0, os.path.getsize(f.name) - 200_000))
            tail = f.read().decode("utf-8", "replace")
        stats["laya_errors"] = tail.count("[LAYA]")
    except OSError:
        pass

    # --- community fund (GA_COMMUNITY): balance + lifetime in/out ---
    stats["fund"] = None
    try:
        with open(f"storage/{sim_code}/resources/economy_v2.json") as f:
            fund = json.load(f).get("fund")
        if fund:
            stats["fund"] = {k: fund.get(k, 0.0) for k in ("balance", "in", "out")}
    except (IOError, ValueError):
        pass

    # --- sales: recent ticker + per-venue intake/till (economy v2 ledger) ---
    stats["sales"] = None
    try:
        with open(f"storage/{sim_code}/resources/economy_v2.json") as f:
            eco = json.load(f)
        log = eco.get("pay_log", [])
        taxes = [x for x in log if x.get("kind") == "tax" and x.get("who") == "sales tax"]
        recent = [x for x in log if x.get("kind") == "sale"][-12:]
        venues = {}
        names = {"cafe": "Hobbs Cafe", "pub": "The Rose and Crown Pub",
                 "grocery": "Harvey Oak Supply Store",
                 "market": "The Willows Market and Pharmacy"}
        for vid, label in names.items():
            venues[label] = {"today": eco.get("daily_intake", {}).get(vid, 0.0),
                             "yesterday": eco.get("yesterday_intake", {}).get(vid, 0.0),
                             "till": eco.get("tills", {}).get(vid, 0.0)}
        stats["sales"] = {
            "recent": [{"buyer": x.get("who"), "venue": x.get("venue"),
                        "item": x.get("note"), "price": x.get("amount"),
                        "t": x.get("t")} for x in reversed(recent)],
            "venues": venues,
            "recent_tax": round(sum(x.get("amount", 0) for x in taxes), 2),
        }
    except (IOError, ValueError):
        pass

    return JsonResponse(stats)


# ---------------------------------------------------------------------------
# VILLE DOTS — per-persona simulated PC (workspace) viewer
# ---------------------------------------------------------------------------

_PC_SAFE_NAME = re.compile(r"^[\w][\w .'\-]*$")  # persona names: "Tamara Taylor"
_PC_SAFE_FILE = re.compile(r"^[\w][\w.\-]*$")    # workspace filenames


def _pc_workspace_dir(sim_code, persona_name):
    """Validated path to a persona's workspace dir, or None."""
    if not _PC_SAFE_NAME.match(persona_name or ""):
        return None
    return os.path.join(
        PROJECT_ROOT, "environment", "frontend_server", "storage",
        sim_code, "personas", persona_name, "workspace")


def get_pc_workspace(request, sim_code, persona_name):
    """JSON API: a persona's whole PC — files, projects, sessions, content."""
    ws_dir = _pc_workspace_dir(sim_code, persona_name)
    if not ws_dir or not os.path.isdir(ws_dir):
        return JsonResponse({"error": "no workspace for this persona"}, status=404)
    try:
        with open(os.path.join(ws_dir, "index.json")) as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        index = {"files": {}, "sessions": []}
    files = [
        {"name": n, **(meta or {})} for n, meta in (index.get("files") or {}).items()
    ]
    current = request.GET.get("file")
    content = None
    status = None
    revisions = []
    if current and _PC_SAFE_FILE.match(current):
        path = os.path.join(ws_dir, "files", current)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                content = None
            status = (index["files"].get(current) or {}).get("status")
            rev_files = sorted(
                (fn for fn in os.listdir(os.path.join(ws_dir, "files"))
                 if fn.startswith(current[:-4] + ".r")),
            )
            revisions = rev_files
    # active projects from the persona's scratch (checkpoint state)
    projects = []
    scratch_path = os.path.join(
        PROJECT_ROOT, "environment", "frontend_server", "storage",
        sim_code, "personas", persona_name, "bootstrap_memory", "scratch.json")
    try:
        with open(scratch_path) as f:
            projects = json.load(f).get("active_projects", [])
    except (OSError, json.JSONDecodeError):
        pass
    return JsonResponse({
        "sim": sim_code, "persona": persona_name,
        "files": files, "projects": projects,
        "sessions": (index.get("sessions") or [])[-30:],
        "current_file": current, "content": content,
        "current_status": status, "revisions": revisions,
    })


def pc_viewer(request, sim_code, persona_name):
    """HTML page: 'open the dot's computer' — file tree, doc viewer,
    revision picker, session activity log."""
    ws_dir = _pc_workspace_dir(sim_code, persona_name)
    if not ws_dir:
        return HttpResponse("invalid persona name", status=400)
    error = None
    index = {"files": {}, "sessions": []}
    if os.path.isdir(ws_dir):
        try:
            with open(os.path.join(ws_dir, "index.json")) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    else:
        error = (f"{persona_name} has no workspace yet — no work sessions "
                 f"have fired in this sim.")
    files = [
        {"name": n, **(meta or {})} for n, meta in (index.get("files") or {}).items()
    ]
    current = request.GET.get("file")
    if current and not _PC_SAFE_FILE.match(current):
        current = None
    content = None
    current_status = None
    revisions = []
    if current and os.path.isdir(os.path.join(ws_dir, "files")):
        path = os.path.join(ws_dir, "files", current)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                current_status = (index["files"].get(current) or {}).get("status")
                revisions = sorted(
                    fn for fn in os.listdir(os.path.join(ws_dir, "files"))
                    if fn.startswith(current[:-4] + ".r"))
            except OSError:
                pass
    projects = []
    scratch_path = os.path.join(
        PROJECT_ROOT, "environment", "frontend_server", "storage",
        sim_code, "personas", persona_name, "bootstrap_memory", "scratch.json")
    try:
        with open(scratch_path) as f:
            projects = json.load(f).get("active_projects", [])
    except (OSError, json.JSONDecodeError):
        pass
    context = {
        "sim_code": sim_code, "persona_name": persona_name,
        "files": files, "projects": projects,
        "sessions": (index.get("sessions") or [])[-15:][::-1],
        "current_file": current, "content": content,
        "current_status": current_status, "revisions": revisions,
        "error": error,
    }
    return render(request, "pc/pc_viewer.html", context)


def pc_note(request, sim_code, persona_name):
    """Ville Dots Phase 4: 'leave a note' — POST a feedback message to the
    persona's workspace inbox. The BACKEND ingests it into associative memory
    in-step; this view only appends to a workspace file (single-writer rule).
    POST-only; text length caps enforced here AND in the store."""
    if request.method != "POST":
        return HttpResponse("POST only", status=405)
    ws_dir = _pc_workspace_dir(sim_code, persona_name)
    if not ws_dir or not os.path.isdir(ws_dir):
        return HttpResponse("no workspace for this persona", status=404)
    text = (request.POST.get("text") or "").strip()[:600]
    author = (request.POST.get("author") or "a reader").strip()[:40]
    if not text:
        return HttpResponse("empty note", status=400)
    # sys.path juggling: reuse the backend's WorkspaceStore exactly as the sim
    # does, so file formats can never drift between frontend and backend.
    import sys as _sys
    _BS = os.path.join(PROJECT_ROOT, "reverie", "backend_server")
    if _BS not in _sys.path:
        _sys.path.insert(0, _BS)
    from ville_os import WorkspaceStore
    store = WorkspaceStore(ws_dir)
    store.push_note(author, text)
    return redirect(f"/pc/{sim_code}/{persona_name}/?note=sent")


def dots_index(request, sim_code):
    """Ville Dots: index of every persona in this sim — with computers listed
    first (files/projects/last session) and idle agents after (no workspace
    yet — appears once their first PC work session fires)."""
    personas = []
    storage = os.path.join(
        PROJECT_ROOT, "environment", "frontend_server", "storage", sim_code,
        "personas")
    if os.path.isdir(storage):
        for name in sorted(os.listdir(storage)):
            scratch_path = os.path.join(storage, name, "bootstrap_memory", "scratch.json")
            if not os.path.isfile(scratch_path):
                continue
            projects = []
            try:
                with open(scratch_path) as f:
                    projects = json.load(f).get("active_projects", [])
            except (OSError, json.JSONDecodeError):
                pass
            entry = {
                "name": name,
                "files": 0, "chars": 0,
                "projects": projects,
                "last_session": None, "last_op": None,
                "has_pc": False,
            }
            index_path = os.path.join(storage, name, "workspace", "index.json")
            if os.path.isfile(index_path):
                try:
                    with open(index_path) as f:
                        index = json.load(f)
                    files = index.get("files") or {}
                    sessions = index.get("sessions") or []
                    entry.update({
                        "has_pc": True,
                        "files": len(files),
                        "chars": sum((m or {}).get("chars", 0) for m in files.values()),
                        "last_session": sessions[-1]["sim_time"] if sessions else None,
                        "last_op": sessions[-1]["op"] if sessions else None,
                    })
                except (OSError, json.JSONDecodeError):
                    pass
            personas.append(entry)
    # computers first, then idle agents
    personas.sort(key=lambda p: (not p["has_pc"], p["name"]))
    context = {"sim_code": sim_code, "personas": personas}
    return render(request, "pc/dots_index.html", context)







