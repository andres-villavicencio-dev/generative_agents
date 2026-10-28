"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reverie.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "personas" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

# FIX (vector 1 — parallel cognition): the step loop ran all 15 personas'
# perceive→retrieve→plan serially, and each persona's cognition is
# independent except for conversations (which mutate BOTH personas' scratch).
# Pool the per-persona move() calls; conversations serialize on CONVO_LOCK
# (shared object lives in concurrency_utils.py so plan.py can import it too).
from concurrency_utils import CONVO_LOCK
# Chat-time accounting: conversations charge the world clock per reply.
import chat_time as _chat_time

# LLM concurrency: 3-4 decode slots fit the 8GB 3070 with the 4b lane model
# (q8_0 KV cache). Threads beyond this just queue at the daemon.
GA_MOVE_WORKERS = int(os.environ.get("GA_MOVE_WORKERS", "6"))
EATING_KEYWORDS = ("eating", "breakfast", "lunch", "dinner", "meal")
MEAL_COOLDOWN_MIN = int(os.environ.get("GA_MEAL_COOLDOWN_MIN", "90"))

# from selenium import webdriver  # not needed for local run

from global_methods import *
from utils import *
from maze import *
from persona.persona import *
from persona.prompt_template.gpt_structure import set_persona_call_context
from resource_manager import WorldResourceManager, ACTION_RESOURCE_MAPPINGS, PRODUCTION_MAPPINGS, get_persona_decay_multiplier, match_production
from artifact_manager import (ArtifactManager, ARTIFACT_EMOJI, ARTIFACT_TYPES,
                              CREATION_ACTION_PATTERNS)
from ville_os import (WorkspaceStore, run_work_session, matches_pc_session)
from chronicle import chronicle_milestone

##############################################################################
#                                  REVERIE                                   #
##############################################################################

class ReverieServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code):
    # FORKING FROM A PRIOR SIMULATION:
    # <fork_sim_code> indicates the simulation we are forking from. 
    # Interestingly, all simulations must be forked from some initial 
    # simulation, where the first simulation is "hand-crafted".
    self.fork_sim_code = fork_sim_code
    fork_folder = f"{fs_storage}/{self.fork_sim_code}"

    # <sim_code> indicates our current simulation. The first step here is to 
    # copy everything that's in <fork_sim_code>, but edit its 
    # reverie/meta/json's fork variable. 
    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"
    # GA_FORK_SKIP_COPY=1: the operator pre-built <sim_code> outside the
    # backend (e.g. a roster-trimmed fork with persona_names edited before
    # boot); continue that storage as-is instead of re-copying the origin.
    # Fail-open: if the pre-built folder is missing/incomplete, copy anyway.
    if (os.environ.get("GA_FORK_SKIP_COPY") == "1"
        and os.path.isfile(f"{sim_folder}/reverie/meta.json")):
      print(f"[Fork] GA_FORK_SKIP_COPY=1 — continuing pre-built {sim_code}")
    else:
      copyanything(fork_folder, sim_folder)

    with open(f"{sim_folder}/reverie/meta.json") as json_file:  
      reverie_meta = json.load(json_file)

    with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
      reverie_meta["fork_sim_code"] = fork_sim_code
      outfile.write(json.dumps(reverie_meta, indent=2))

    # LOADING REVERIE'S GLOBAL VARIABLES
    # The start datetime of the Reverie: 
    # <start_datetime> is the datetime instance for the start datetime of 
    # the Reverie instance. Once it is set, this is not really meant to 
    # change. It takes a string date in the following example form: 
    # "June 25, 2022"
    # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
    self.start_time = datetime.datetime.strptime(
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <maze> is the main Maze instance. Note that we pass in the maze_name
    # (e.g., "double_studio") to instantiate Maze. 
    # e.g., Maze("double_studio")
    self.maze = Maze(reverie_meta['maze_name'])
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our personas made in terms
    # of the number of tiles. 
    # RESUME CHECKPOINT OVERRIDE: when resuming a sim from a checkpoint step
    # beyond the (possibly stale) fork meta, GA_RESUME_STEP / GA_RESUME_TIME
    # force the sim to start at that step. Used by run_live3d_resume.py, whose
    # fork parent's meta.json step lags the actual checkpoint.
    self.step = reverie_meta['step']
    resume_step = os.environ.get("GA_RESUME_STEP")
    if resume_step is not None:
      self.step = int(resume_step)
      resume_time = os.environ.get("GA_RESUME_TIME")
      if resume_time:
        self.curr_time = datetime.datetime.strptime(
            resume_time, "%B %d, %Y, %H:%M:%S")
        reverie_meta['curr_time'] = resume_time
        with open(f"{sim_folder}/reverie/meta.json", "w") as outfile:
          outfile.write(json.dumps(reverie_meta, indent=2))
      print(f"[reverie] resume checkpoint override: step={self.step} "
            f"curr_time={self.curr_time}")

    # Clean up stale environment and movement files from the parent sim
    # that are beyond our starting step. These cause race conditions
    # between the frontend and backend if left in place: the frontend
    # reads parent's pre-existing movement files before the backend can
    # overwrite them with fresh computations, causing agents to follow
    # the parent's historical path and visually teleport.
    for subfolder in ["environment", "movement"]:
      cleanup_folder = f"{sim_folder}/{subfolder}"
      if os.path.exists(cleanup_folder):
        for fname in os.listdir(cleanup_folder):
          if fname.endswith(".json"):
            try:
              file_step = int(fname.split(".")[0])
              if file_step > self.step:
                os.remove(os.path.join(cleanup_folder, fname))
            except ValueError:
              pass
    # Boot-race sync point (Oct 2026): run_live3d_resume waits for this line
    # before starting the headless driver — starting it earlier let the
    # driver's fresh environment/<step+1>.json land BEFORE this cleanup and
    # get deleted, wedging the fork in a silent mutual deadlock.
    print("[reverie] stale-file cleanup complete — driver may start")

    # SETTING UP PERSONAS IN REVERIE
    # <personas> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all personas who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.personas = dict()
    # <personas_tile> is a dictionary that contains the tile location of
    # the personas (!-> NOT px tile, but the actual tile coordinate).
    # The tile take the form of a set, (row, col). 
    # e.g., ["Isabella Rodriguez"] = (58, 39)
    self.personas_tile = dict()
    
    # # <persona_convo_match> is a dictionary that describes which of the two
    # # personas are talking to each other. It takes a key of a persona's full
    # # name, and value of another persona's full name who is talking to the 
    # # original persona. 
    # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
    # self.persona_convo_match = dict()
    # # <persona_convo> contains the actual content of the conversations. It
    # # takes as keys, a pair of persona names, and val of a string convo. 
    # # Note that the key pairs are *ordered alphabetically*. 
    # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
    # self.persona_convo = dict()

    # Loading in all personas. 
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]
      curr_persona = Persona(persona_name, persona_folder)

      curr_persona.scratch.sim_folder = sim_folder
      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

    # COMMUNITY BULLETIN BOARD
    # Register a bulletin board object in Johnson Park for death notices
    # and other community announcements.
    self.bulletin_tile = None
    park_address = "the Ville:Johnson Park:park"
    if park_address in self.maze.address_tiles:
      self.bulletin_tile = list(self.maze.address_tiles[park_address])[0]
      bulletin_addr = f"{park_address}:community bulletin board"
      self.maze.address_tiles[bulletin_addr] = {self.bulletin_tile}
      bx, by = self.bulletin_tile
      self.maze.tiles[by][bx]["events"].add(
        ("community bulletin board", None, None, None))
      # Add to all personas' spatial memory
      for p in self.personas.values():
        tree = p.s_mem.tree
        if "the Ville" in tree:
          if "Johnson Park" in tree["the Ville"]:
            if "park" in tree["the Ville"]["Johnson Park"]:
              arenas = tree["the Ville"]["Johnson Park"]["park"]
              if "community bulletin board" not in arenas:
                arenas.append("community bulletin board")

    # REVERIE SETTINGS PARAMETERS:
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine.
    self.server_sleep = 0.1

    # SIGNALING THE FRONTEND SERVER: 
    # curr_sim_code.json contains the current simulation code, and
    # curr_step.json contains the current step of the simulation. These are 
    # used to communicate the code and step information to the frontend. 
    # Note that step file is removed as soon as the frontend opens up the 
    # simulation. 
    curr_sim_code = dict()
    curr_sim_code["sim_code"] = self.sim_code
    with open(f"{fs_temp_storage}/curr_sim_code.json", "w") as outfile: 
      outfile.write(json.dumps(curr_sim_code, indent=2))
    
    curr_step = dict()
    curr_step["step"] = self.step
    with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile:
      outfile.write(json.dumps(curr_step, indent=2))

    # Initialize the world resource manager (Phase 2)
    self.resource_manager = WorldResourceManager(sim_folder)
    # Attach to maze for easy access throughout the cognitive modules
    self.maze.resource_manager = self.resource_manager

    self.resource_manager.ensure_home_fridges(self.personas)

    # Economy v2 (Oct 2026): the 25-agent town economy — per-venue tills,
    # owner draws, restock job, student stipends, LLM hiring market.
    # Gate GA_ECONOMY2 (default ON; =0 opts out). Fail-open everywhere.
    self.economy = None
    if os.environ.get("GA_ECONOMY2", "1") != "0":
      try:
        from economy import EconomyManager
        from persona.prompt_template.gpt_structure import ChatGPT_single_request
        self.economy = EconomyManager(sim_folder)
        # the delivery path needs the economy for wholesale routing
        self.resource_manager.economy = self.economy
        def _econ_llm(prompt):
          return ChatGPT_single_request(prompt) or ""
        def _econ_bulletin(text):
          # Post to the community board the same way death notices do.
          if self.bulletin_tile:
            bx, by = self.bulletin_tile
            self.maze.tiles[by][bx]["events"].add(
              ("community bulletin board", "announces", text, text))
        self.economy.attach(
          llm=_econ_llm,
          personas=self.personas,
          bulletin=_econ_bulletin)
        from persona.cognitive_modules import plan as _plan_mod
        _plan_mod._ECON_REF["e"] = self.economy
        print(f"[Economy] v2 active: {self.economy}")
      except Exception as _econ_err:
        print(f"[Economy] v2 init failed ({_econ_err.__class__.__name__}) "
              f"— continuing without economy")
        self.economy = None

    # Initialize the artifact manager (Phase 6: Artifact Creation)
    self.artifact_manager = ArtifactManager(sim_folder)
    self.maze.artifact_manager = self.artifact_manager
    # VILLE DOTS: per-persona simulated PC workspaces. Lazily created on first
    # session (a fresh sim has no workspace dirs until someone works).
    self.sim_folder = sim_folder
    self._workspace_stores = {}
    self._pc_session_counter = {}   # per-agent: matching actions since last session
    # Rate guard: fire a session on every Nth matching action TRANSITION.
    # Default 1 — transitions are rare (an agent enters a given writing action
    # once per sim-day; the 100+ step occurrences of one description collapse
    # to a single transition), so every=3 starved sessions in n47: 7 agents
    # matched 945 times but only Tamara hit 3 transitions. Raise only if the
    # artifact lane ever shows load.
    self._pc_session_every = int(os.environ.get("GA_PC_SESSION_EVERY", "1"))
    # Restore artifact events on maze tiles from saved state
    self._restore_artifact_tiles()


  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the personas.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # Wait for background conversations to be written back first — a
    # checkpoint must not serialize a half-updated persona.
    try:
      from concurrency_utils import join_chat_threads
      join_chat_threads()
    except Exception as e:
      print(f"[Reverie] Warning: chat-thread join before save failed: {e}")
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.personas.keys())
    reverie_meta["step"] = self.step
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w") as outfile: 
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the personas.
    for persona_name, persona in self.personas.items():
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.save(save_folder)

    # Save the world resource state (Phase 2)
    try:
      self.resource_manager.save(sim_folder)
    except Exception as e:
      print(f"[Reverie] Warning: Could not save resource manager: {e}")

    # Save artifact state (Phase 6)
    try:
      self.artifact_manager.save(sim_folder)
    except Exception as e:
      print(f"[Reverie] Warning: Could not save artifact manager: {e}")


  def _restore_artifact_tiles(self):
    """Re-place all active artifacts onto maze tiles after restart."""
    for art in self.artifact_manager.get_all_active_artifacts():
      location = art["location"]
      if location in self.maze.address_tiles:
        tile = list(self.maze.address_tiles[location])[0]
        event_tuple = self.artifact_manager.get_artifact_event_tuple(art["id"])
        if event_tuple:
          tx, ty = tile
          self.maze.tiles[ty][tx]["events"].add(event_tuple)

  def try_create_artifact(self, persona, act_description):
    """Check if an action creates an artifact and handle creation."""
    # FIX (accumulation gate): task decomposition shreds creative blocks into
    # 5-30 min subtasks, so no single action ever met the old hard
    # duration_min gate — artifacts were NEVER created because of this.
    # Now accumulate minutes per artifact type per sim day; create when the
    # day's total effort reaches duration_min.
    # NOTE: identical consecutive subtask descriptions are real effort and
    # must count — the old dedup guard starved accumulation. Rate control is
    # the accumulation reset itself: one artifact per duration_min of work.
    mapping = self.artifact_manager.match_creation_action(act_description)
    if not mapping:
      return None

    duration_min = mapping.get("duration_min", 0)
    act_duration = persona.scratch.act_duration or 0
    day_key = self.curr_time.strftime("%Y-%m-%d")
    prog_key = f"{mapping['type']}|{day_key}"
    if not hasattr(persona.scratch, "artifact_progress"):
      persona.scratch.artifact_progress = {}
    # PERF (improvement 3a — effort credit): task decomposition shreds creative
    # blocks into 5-30 min subtasks, so raw minutes understate real effort —
    # n30 logged 66 creative actions but produced only 2 artifacts in 1.7 days.
    # Credit x2 keeps the accumulation gate meaningful (book=60min now really
    # means ~30 real minutes of scheduled work).
    persona.scratch.artifact_progress[prog_key] = (
      persona.scratch.artifact_progress.get(prog_key, 0) + 2 * max(act_duration, 5))
    if persona.scratch.artifact_progress[prog_key] < duration_min:
      return None  # not enough accumulated effort yet today
    # Threshold met — create and reset this key's progress
    persona.scratch.artifact_progress[prog_key] = 0

    # Consume materials if required
    materials = mapping.get("materials", {})
    if materials and hasattr(self, 'resource_manager'):
      for item, amount in materials.items():
        # Try to consume from agent's current location
        if persona.scratch.act_address:
          self.resource_manager.consume(persona.scratch.act_address, item, amount)

    # Generate artifact properties via LLM
    from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_create_artifact
    artifact_props = run_gpt_prompt_create_artifact(
      persona, mapping["type"], act_description)

    # Create the artifact
    created_time = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    location = persona.scratch.act_address or "unknown"

    artifact = self.artifact_manager.create_artifact(
      artifact_type=mapping["type"],
      name=artifact_props["name"],
      description=artifact_props["description"],
      content_summary=artifact_props["content_summary"],
      creator=persona.name,
      created_time=created_time,
      quality=artifact_props["quality"],
      location=location,
      materials_consumed=materials if materials else None,
    )

    # Generate full artifact content via LLM
    try:
      from artifact_generator import generate_artifact_content
      persona_desc = persona.scratch.get_str_iss()
      content_full, content_type = generate_artifact_content(
        persona.name, act_description, persona_desc)
      if content_full:
        artifact["content_full"] = content_full
        print(f"[Reverie] Generated {content_type} content for {artifact['name']} ({len(content_full)} chars)")
    except Exception as e:
      print(f"[Reverie] Warning: artifact content generation failed: {e}")

    # Place artifact on maze tile
    if location in self.maze.address_tiles:
      tile = list(self.maze.address_tiles[location])[0]
      event_tuple = self.artifact_manager.get_artifact_event_tuple(artifact["id"])
      if event_tuple:
        tx, ty = tile
        self.maze.tiles[ty][tx]["events"].add(event_tuple)

    # Add creation event to persona's memory
    from persona.prompt_template.gpt_structure import get_embedding
    desc = f"{persona.name} created {artifact['name']}, a {artifact['type']}"
    keywords = set([artifact["type"], artifact["name"].lower(), "created"])
    if desc in persona.a_mem.embeddings:
      embedding = persona.a_mem.embeddings[desc]
    else:
      embedding = get_embedding(desc)
    persona.a_mem.add_event(
      persona.scratch.curr_time, None,
      persona.name, "created", artifact["name"],
      desc, keywords, 6, (desc, embedding), [])

    # Mark dedup guard
    persona.scratch.last_artifact_action = act_description

    # Bulletin board announcement for high quality artifacts
    if artifact["quality"] >= 7 and self.bulletin_tile:
      bx, by = self.bulletin_tile
      emoji = ARTIFACT_EMOJI.get(artifact["type"], "📦")
      announcement = (
        "community bulletin board", "announces",
        f"New {artifact['type']}: '{artifact['name']}' by {persona.name}",
        f"{emoji} New {artifact['type']}: '{artifact['name']}' by {persona.name}")
      self.maze.tiles[by][bx]["events"].add(announcement)

    print(f"[Reverie] {persona.name} created artifact: {artifact['name']} ({artifact['type']}, quality={artifact['quality']})")
    return artifact

  def _get_workspace(self, persona_name):
    """Ville Dots: lazily get (or create) a persona's WorkspaceStore."""
    if persona_name not in self._workspace_stores:
      ws_dir = os.path.join(self.sim_folder, "personas", persona_name, "workspace")
      self._workspace_stores[persona_name] = WorkspaceStore(ws_dir)
    return self._workspace_stores[persona_name]

  def _drain_pc_notes_for(self, persona):
    """Ville Dots Phase 4: ingest pending external feedback notes (from the
    PC viewer's 'leave a note' form) into the persona's associative memory
    through the REAL memory path — real schema, real embedding. In-step only
    (the frontend only appends to a workspace file; never touches memory)."""
    store = self._get_workspace(persona.name)
    notes = store.drain_notes()
    if not notes:
      return
    from persona.prompt_template.gpt_structure import get_embedding
    for note in notes:
      author = (note.get("author") or "a reader").strip() or "a reader"
      text = note.get("text", "").strip()
      if not text:
        continue
      desc = f"{author} left a note for {persona.name} about their work: {text}"
      keywords = set(["note", "feedback", author.lower()])
      if desc in persona.a_mem.embeddings:
        embedding = persona.a_mem.embeddings[desc]
      else:
        embedding = get_embedding(desc)
      persona.a_mem.add_event(
        persona.scratch.curr_time, None,
        persona.name, "received", "a note about their work",
        desc, keywords, 6, (desc, embedding), [])
      print(f"[VilleOS] {persona.name} ingested a feedback note from {author}: "
            f"\"{text[:60]}\"")
    return len(notes)

  def _run_pc_session_for(self, persona, act_description):
    """Ville Dots: run one bounded work session against the persona's
    simulated PC (workspace) when their action is project work.
    Rate-guarded: only every GA_PC_SESSION_EVERY-th matching action fires
    an LLM session so step rate isn't taxed by every 5-30min subtask."""
    try:
      name = persona.name
      counter = self._pc_session_counter.get(name, 0) + 1
      self._pc_session_counter[name] = counter
      if counter < self._pc_session_every:
        return None
      self._pc_session_counter[name] = 0
      # Pick or seed the persona's current project.
      scratch = persona.scratch
      projects = getattr(scratch, "active_projects", None) or []
      project = None
      for p in projects:
        if p.get("status") in (None, "drafting", "revising"):
          project = p
          break
      if project is None:
        # Seed a new project from the action itself (identity-driven: the
        # action already names the work — e.g. Tamara's novel series).
        # Strip leading motion verbs so the title reads like a work, not
        # an action ("continue writing X" -> "X").
        import re as _re
        raw_title = act_description.split("@")[0].strip()[:80]
        # strip trailing task-decomp parentheticals BEFORE verb cleanup —
        # decomp annotations like "begin writing X (X with courage)" leak
        # the paren text into the title otherwise
        raw_title = _re.sub(r"\s*\(.*$", "", raw_title).strip()
        raw_title = _re.sub(
          r"^(continue|resume|begin|start|finish|work on|working on)\s+"
          r"(the\s+)?(writing|drafting|focused writing)?\s*", "", raw_title,
          flags=_re.IGNORECASE).strip(" .,:()-") or raw_title
        slug = _re.sub(r"[^a-z0-9]+", "_", raw_title.lower()).strip("_")[:30]
        project = {
          "title": raw_title[:60] or "untitled",
          "type": "book", "file": (slug + ".txt") if slug else "",
          "status": "drafting",
          "sessions_done": 0, "target_revs": 8,
        }
        scratch.active_projects.append(project)
      project["sessions_done"] = project.get("sessions_done", 0) + 1
      store = self._get_workspace(name)

      def model_call(prompt):
        from persona.prompt_template.gpt_structure import ChatGPT_request
        return ChatGPT_request(prompt, model_hint="artifact")

      result = run_work_session(
        store, name, project, act_description, model_call,
        sim_time=self.curr_time.strftime("%B %d, %Y, %H:%M:%S"))
      if result:
        op, file_name = result
        print(f"[VilleOS] {name} session: {op} {file_name} "
              f"(project '{project['title']}' session {project['sessions_done']})")
        # A finished project feeds the existing artifact pipeline: promote
        # the workspace file into a real artifact the town can see.
        if op == "FINISH":
          scratch.active_projects = [
            p for p in scratch.active_projects if p is not project] + [project]
          try:
            # The workspace file IS the finished work — promote its actual
            # text (never re-generate; the project's evolution is the artifact).
            content = store.read(file_name)
            if content:
              artifact = self.artifact_manager.create_artifact(
                artifact_type=project.get("type", "book"),
                name=project["title"],
                description=f"Finished project by {persona.name}",
                content_summary=content[:200],
                creator=persona.name,
                created_time=self.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
                quality=7,
                location=persona.scratch.act_address or "unknown",
                materials_consumed=None,
              )
              artifact["content_full"] = content
              print(f"[VilleOS] {name} promoted finished project to artifact: "
                    f"{artifact['name']}")
          except Exception as e:
            print(f"[VilleOS] Warning: artifact promotion failed: {e}")
      return result
    except Exception as e:
      print(f"[VilleOS] Warning: PC session failed: {e}")
      return None

  def process_artifact_interaction(self, persona, act_description):
    """Check if an action involves interacting with an artifact."""
    matching_types = self.artifact_manager.match_interaction(act_description)
    if not matching_types:
      return

    # Find artifacts at the persona's current location
    location = persona.scratch.act_address
    if not location:
      return

    artifacts = self.artifact_manager.get_artifacts_at_location(location)
    if not artifacts:
      # Try parent address (without game object)
      parts = location.split(":")
      for i in range(len(parts) - 1, 0, -1):
        parent = ":".join(parts[:i])
        artifacts = self.artifact_manager.get_artifacts_at_location(parent)
        if artifacts:
          break

    if not artifacts:
      return

    # Find the best matching artifact
    target = None
    for art in artifacts:
      if art["type"] in matching_types:
        target = art
        break

    if not target:
      return

    # Handle consumable artifacts (meals)
    if target.get("is_consumable"):
      self.artifact_manager.consume_artifact(target["id"], persona.name)
      # Remove from maze tile
      event_tuple = self.artifact_manager.get_artifact_event_tuple(target["id"])
      if event_tuple and target["location"] in self.maze.address_tiles:
        tile = list(self.maze.address_tiles[target["location"]])[0]
        tx, ty = tile
        self.maze.tiles[ty][tx]["events"].discard(event_tuple)

    # Apply need effects
    type_info = ARTIFACT_TYPES.get(target["type"], {})
    need_effects = type_info.get("need_effect", {})
    if hasattr(persona.scratch, "needs"):
      for need, boost in need_effects.items():
        if need in persona.scratch.needs:
          persona.scratch.needs[need] = min(100, persona.scratch.needs[need] + boost)

    # Generate thought reaction via LLM
    from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_artifact_reaction
    from persona.prompt_template.gpt_structure import get_embedding
    thought = run_gpt_prompt_artifact_reaction(persona, target)

    keywords = set([target["type"], target["name"].lower(), target["creator"].lower()])
    if thought in persona.a_mem.embeddings:
      embedding = persona.a_mem.embeddings[thought]
    else:
      embedding = get_embedding(thought)
    persona.a_mem.add_thought(
      persona.scratch.curr_time, None,
      persona.name, "thinks about", target["name"],
      thought, keywords, 4, (thought, embedding), [])

  def start_path_tester_server(self):
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def _night_step_seconds(self):
    """Night time dilation (GA_NIGHT_DILATION=<factor>, 0/1 = off).

    While EVERY living persona is in an unfinished sleep action, nothing
    can change until someone wakes, so each step spans `factor` x
    sec_per_step of sim time instead of sec_per_step. Guards: the new-day
    planning burst has run (hour >= GA_NIGHT_START_HOUR and every persona's
    scratch clock is on today's date), no persona is mid-chat, and no
    reflection is due. The jump never crosses (earliest action end -
    GA_NIGHT_WAKE_MARGIN_MIN), so the town is back at normal speed an hour
    before the first waker. Any non-sleep action ends dilation. Step files
    are still written every step, so viewer/checkpoints are unaffected.
    """
    base = self.sec_per_step
    try:
      factor = float(os.environ.get("GA_NIGHT_DILATION", "0"))
    except ValueError:
      factor = 0.0
    if factor <= 1.0:
      return base
    try:
      from concurrency_utils import PENDING_CHATS, ASYNC_FROZEN, CHAT_BUSY
      if PENDING_CHATS or ASYNC_FROZEN or CHAT_BUSY:
        return base
      start_h = int(os.environ.get("GA_NIGHT_START_HOUR", "1"))
      if not (start_h <= self.curr_time.hour < 12):
        return base
      margin = datetime.timedelta(
        minutes=int(os.environ.get("GA_NIGHT_WAKE_MARGIN_MIN", "60")))
      earliest_end = None
      living = 0
      for persona in self.personas.values():
        s = persona.scratch
        if getattr(s, "is_deceased", False):
          continue
        living += 1
        desc = (s.act_description or "").lower()
        if not ("sleep" in desc or "asleep" in desc or "in bed" in desc):
          return base
        if s.chatting_with or not s.act_address or s.act_check_finished():
          return base
        if (not s.curr_time
            or s.curr_time.date() != self.curr_time.date()):
          return base
        if s.importance_trigger_curr <= 0:
          return base
        x = s.act_start_time
        if x.second != 0:
          x = x.replace(second=0) + datetime.timedelta(minutes=1)
        end = x + datetime.timedelta(minutes=s.act_duration)
        if earliest_end is None or end < earliest_end:
          earliest_end = end
      if not living or earliest_end is None:
        return base
      room = (earliest_end - margin - self.curr_time).total_seconds()
      step_s = int(base * factor)
      if room < step_s:
        return base
      return step_s
    except Exception as e:
      print(f"[NightDilation] disabled this step ({e.__class__.__name__}: {e})")
      return base

  def tick_needs(self, dt_scale=1.0):
    """
    Decay agent needs each simulation tick.
    Called from the main simulation loop after each step.

    Behavior:
    - Loop over all personas
    - Check if agent is sleeping (from act_description)
    - Decay each need by its rate (with persona-specific multiplier) each tick
    - Energy restores during sleep (+1/60 per tick), decays when awake
    - Social only decays when not chatting_with anyone
    - Hunger/hydration/bladder decay at 30% rate during sleep
    - Clamp all values 0-100
    """
    for persona_name, persona in self.personas.items():
      s = persona.scratch

      # Guard for backwards compatibility
      if not hasattr(s, "needs"):
        continue

      # Check if sleeping
      is_sleeping = False
      if s.act_description:
        act_lower = s.act_description.lower()
        is_sleeping = "sleep" in act_lower or "asleep" in act_lower or "in bed" in act_lower

      # Check if chatting
      is_chatting = s.chatting_with is not None

      for need, base_rate in s.needs_decay_rates.items():
        current_val = s.needs[need]

        # Apply persona-specific multiplier
        multiplier = get_persona_decay_multiplier(persona_name, need)
        rate = base_rate * multiplier

        if need == "energy":
          if is_sleeping:
            # Energy restores during sleep
            current_val += 1/60
          else:
            # Energy decays when awake
            current_val -= rate
        elif need == "social":
          # Social only decays when not chatting
          if not is_chatting:
            current_val -= rate
        elif need in ("hunger", "hydration", "bladder"):
          # These decay at 30% rate during sleep
          if is_sleeping:
            current_val -= rate * 0.3
          else:
            current_val -= rate
        else:
          # hygiene, comfort, stimulation decay normally (but not during sleep for simplicity)
          if not is_sleeping:
            current_val -= rate

        # Night dilation: a step may span dt_scale ticks of sim time.
        if dt_scale != 1.0:
          current_val = s.needs[need] + (current_val - s.needs[need]) * dt_scale

        # Clamp to 0-100
        s.needs[need] = max(0, min(100, current_val))


  def _on_luxury_purchase(self, persona, item):
    """A luxury good satisfies the status need and becomes a memorable
    event (Oct 22 2026). No-op for ordinary items."""
    try:
      from economy import LUXURY_GOODS
      if item not in LUXURY_GOODS:
        return
      s = persona.scratch
      if isinstance(getattr(s, "needs", None), dict):
        s.needs["status"] = min(100, s.needs.get("status", 50) + 70)
      s.last_luxury_purchase = self.curr_time
      price = LUXURY_GOODS[item]
      desc = (f"{persona.name} bought a {item} at The Willows Market for "
              f"${price:.0f} — a reward for saving up, and a sign of how far "
              f"{persona.name.split()[0]} has come.")
      from persona.prompt_template.gpt_structure import get_embedding
      try:
        emb = get_embedding(desc)
      except Exception:
        emb = [0.0] * 768
      persona.a_mem.add_event(
        self.curr_time, None, persona.name, "bought", item, desc,
        {persona.name, item, "luxury", "status"}, 7, (desc, emb), [])
      print(f"[Status] {persona.name} bought a {item} (${price:.0f}); "
            f"status -> {s.needs.get('status', 0):.0f}")
    except Exception as e:
      print(f"[Status] luxury purchase hook failed: {e}")

  def _status_thoughts(self):
    """The status need LINGERS instead of forcing action (Oct 22 2026):
    once per sim-day (from 9 AM), an agent whose status is low gets a
    thought about the luxury good it is saving for — retrieval then brings
    it into planning and conversation. A purchase only happens once the
    agent can afford it (plan.py luxury rule)."""
    try:
      from economy import LUXURY_GOODS, LUXURY_RESERVE
    except Exception:
      return
    if self.curr_time is None or self.curr_time.hour < 9:
      return
    today = self.curr_time.date()
    by_price = sorted(LUXURY_GOODS.items(), key=lambda kv: kv[1])
    for name, persona in self.personas.items():
      s = persona.scratch
      try:
        status = (getattr(s, "needs", None) or {}).get("status", 100)
        if status >= 40 or getattr(s, "last_status_thought_day", None) == today:
          continue
        act = (s.act_description or "").lower()
        if "sleep" in act or s.chatting_with:
          continue
        s.last_status_thought_day = today
        first = name.split()[0]
        wallet = getattr(s, "wallet", 0) or 0
        affordable = [kv for kv in by_price if wallet - kv[1] >= LUXURY_RESERVE]
        if affordable:
          item, price = affordable[-1]
          thought = (f"{first} keeps thinking about the {item} at The Willows "
                     f"Market (${price:.0f}) — with ${wallet:.0f} saved, "
                     f"{first} could finally afford it.")
        else:
          item, price = next(((i, p) for i, p in by_price if p + LUXURY_RESERVE > wallet),
                             by_price[-1])
          thought = (f"{first} keeps thinking about the {item} in The Willows "
                     f"Market window (${price:.0f}); with only ${wallet:.0f} "
                     f"saved, {first} wants to earn more and get ahead.")
        from persona.prompt_template.gpt_structure import get_embedding
        try:
          emb = get_embedding(thought)
        except Exception:
          emb = [0.0] * 768
        persona.a_mem.add_thought(
          self.curr_time, self.curr_time + datetime.timedelta(days=2),
          name, "wants", item, thought, {name, item, "luxury", "status"},
          4, (thought, emb), [])
        print(f"[Status] {name} (status {status:.0f}): {thought}")
      except Exception as e:
        print(f"[Status] thought for {name} failed: {e}")

  def _queue_grocery_run(self, persona):
    """Queue ONE grocery run, honoring the same attempt cooldown (4h) as the
    proactive economy rule in plan.py — this empty-fridge path used to push
    its own differently-worded goal past every dedupe and cooldown."""
    s = persona.scratch
    if any("grocer" in g.lower() for g in s.resource_goals):
      return
    now = s.curr_time
    for stamp in (getattr(s, "last_grocery_attempt", None),
                  getattr(s, "last_grocery_shop", None)):
      try:
        if stamp is not None and (now - stamp).total_seconds() < 4 * 3600:
          return
      except Exception:
        pass
    s.resource_goals.append("buying groceries at the Harvey Oak Supply Store")
    s.last_grocery_attempt = now

  def _read_board_hook(self, persona, act_description):
    """GA_JOBBOARD: an agent that has arrived at the community bulletin board reads the
    openings and may apply; the outcome becomes a memory."""
    try:
      if self.economy is None or "bulletin board" not in (act_description or "").lower():
        return
      import economy as _eco
      if not _eco.jobboard_enabled():
        return
      tile = self.maze.access_tile(persona.scratch.curr_tile)
      if (tile.get("sector") or "").lower() != "johnson park":
        return                                    # only a visit to the park counts as reading it
      desc = self.economy.read_board(persona.name, self.curr_time)
      if not desc:
        return
      from persona.prompt_template.gpt_structure import get_embedding
      try:
        emb = get_embedding(desc)
      except Exception:
        emb = [0.0] * 768
      persona.a_mem.add_event(
        self.curr_time, None, persona.name, "read", "community bulletin board", desc,
        {persona.name, "community bulletin board", "job", "work"},
        9 if ("took the town-service post" in desc or "hired them" in desc) else 5,
        (desc, emb), [])
    except Exception as e:
      print(f"[JobBoard] hook failed: {e}")

  def _arrived(self, persona):
    """True when the persona stands in the sector its current action is addressed to.
    Purchases and consumption used to fire the moment an action began, i.e. while the
    agent was still at home — groceries were bought without going to the store."""
    try:
      addr = persona.scratch.act_address
      if not addr:
        return True
      parts = addr.split(":")
      if len(parts) < 2:
        return True
      tile = self.maze.access_tile(persona.scratch.curr_tile)
      return (tile.get("sector") or "").lower() == parts[1].strip().lower()
    except Exception:
      return True

  def _process_shopping(self, persona, act_description):
    """FIX (shopping loop): when a persona is AT the supply store with a
    grocery goal, actually buy goods — purchase() deducts wallet money and
    moves stock out of store shelves — then carry them home and restock the
    home fridge, and record the purchase in memory."""
    try:
      s = persona.scratch
      curr = (s.act_address or "").lower()
      at_market = "willows market" in curr
      if "supply store" not in curr and "shelves" not in curr and not at_market:
        return  # not at a grocery
      act_lower = (act_description or "").lower()
      if not any(k in act_lower for k in ("grocery", "groceries", "shopping", "buy")):
        return  # not a shopping action
      if "luxury" in act_lower:
        return  # a luxury purchase is not a grocery run

      # Which home fridge does this persona own?
      persona_first = persona.name.split()[0].lower()
      home_fridge = self.resource_manager.home_fridge_of.get(persona.name)
      if not home_fridge:
        for address in self.resource_manager.world_state:
          if "refrigerator" in address.lower() and persona_first in address.lower():
            home_fridge = address
            break
      if not home_fridge:
        return  # no home fridge to stock

      # Pin the CANONICAL store address (STORE_ADDRESSES[0]) — iterating
      # world_state picks whichever duplicate key comes FIRST in the dict
      # (the legacy 'store:shelves:groceries' shadow store), silently
      # draining phantom stock instead of the real one.
      store_addr = None
      if at_market:
        store_addr = next((a for a in self.resource_manager.world_state
                           if "willows market" in a.lower()
                           and "grocery store shelf" in a.lower()), None)
        if not store_addr:
          return
      canonical = self.resource_manager.STORE_ADDRESSES[0]
      norm = None if at_market else self.resource_manager._normalize_address(canonical)
      if norm:
        store_addr = norm
      if not store_addr:
        for address in self.resource_manager.world_state:
          if "supply store" in address.lower() or ("shelves" in address.lower() and "store" in address.lower()):
            store_addr = address
            break
      if not store_addr:
        return

      # Shopping list: staple items below a home-fridge threshold
      staples = ["eggs", "bread", "milk", "coffee_beans"]
      bought = []
      from resource_manager import ITEM_PRICES
      _budget = max(0.0, (getattr(s, "wallet", 0.0) or 0.0)) * 0.35
      _spent = 0.0
      for item in staples:
        home_stock = self.resource_manager.world_state[home_fridge].get(item, 0)
        if home_stock < 5:
          qty = 10  # buy a batch
          # Never spend more than ~35% of the wallet on one stock-up: a $100
          # agent used to put $52 on a single trip and then starve at the till.
          _left = _budget - _spent
          _unit = ITEM_PRICES.get(item, 1.0)
          qty = int(min(qty, _left // _unit))
          if qty < 1:
            continue
          _spent += _unit * qty
          if self.resource_manager.purchase(store_addr, item, qty, s):
            bought.append((item, qty))
            self.resource_manager.restock(home_fridge, item, qty)
            # Money leak fix (Oct 22 2026): under economy v2 purchase() no
            # longer credits any till, and this bulk path never called
            # credit_sale — the wallet was debited and the money vanished.
            if self.economy is not None:
              try:
                _vid = self.economy.venue_of_address(store_addr)
                if _vid is not None:
                  self.economy.credit_sale(
                    _vid, item, qty, persona.name,
                    self.economy.item_price(item) * qty)
              except Exception as _ce:
                print(f"[Economy] bulk-shop credit failed ({_ce.__class__.__name__})")
      if not bought:
        return

      # Remove the fulfilled shopping goal
      if hasattr(s, "resource_goals"):
        s.resource_goals = [g for g in s.resource_goals
                            if "grocer" not in g.lower()]

      # Memory event so the purchase is part of the agent's lived experience
      summary = ", ".join(f"{qty} {item}" for item, qty in bought)
      cost = sum(ITEM_PRICES.get(item, 1.0) * qty for item, qty in bought)
      desc = f"{persona.name} bought {summary} at the supply store for ${cost:.2f} and restocked the refrigerator at home."
      from persona.prompt_template.gpt_structure import get_embedding
      try:
        emb = get_embedding(desc)
      except Exception:
        emb = [0.0] * 1536
      keywords = {"supply store", "groceries", "bought", "restocked"}
      persona.a_mem.add_event(
        self.curr_time, None,
        persona.name, "bought", "groceries",
        desc, keywords, 6, (desc, emb), [])

      print(f"[Shopping] {persona.name} bought {summary} (${cost:.2f}) -> {home_fridge}")
    except Exception as e:
      print(f"[Shopping] Error processing shopping action: {e}")

  def consume_resources_for_action(self, persona, act_description):
    """
    Consume world resources based on the action being performed.
    Called when a persona transitions to a new action.

    If resources are insufficient, injects a perception event into memory
    and triggers a replan.

    Returns:
      bool: True if all resources consumed successfully, False if any failed
    """
    if not act_description:
      return True

    act_lower = act_description.lower()
    persona_name = persona.name
    curr_location = persona.scratch.act_address or ""
    persona_first = persona_name.split()[0].lower()

    all_success = True

    # Find matching action keywords. Longest match wins: "buying groceries
    # at the market" also contains "groceries at the market" and
    # "groceries", and every matching key used to fire — one visit, two or
    # three charges (Oct 22 2026).
    _matched = []
    for keyword, consumptions in sorted(ACTION_RESOURCE_MAPPINGS.items(),
                                        key=lambda kv: -len(kv[0])):
      if keyword not in act_lower:
        continue
      if any(keyword in longer for longer in _matched):
        continue
      _matched.append(keyword)
      # One meal = one consumption. Every 5-minute sub-task of a meal
      # ("have lunch (washing hands)", "(eating the main course)", ...) is a
      # new description and used to consume the whole meal again (~120 meal
      # transitions per agent in 6 sim-hours, which emptied the fridges); a
      # failed meal also replanned every step, so agents shuffled between
      # home and the cafe. Cooldown per agent, stamped on the ATTEMPT.
      if keyword in EATING_KEYWORDS:
        _now = self.curr_time
        _last = getattr(persona.scratch, "last_meal_consumed", None)
        if (_last is not None and _now is not None
            and (_now - _last).total_seconds() < MEAL_COOLDOWN_MIN * 60):
          continue
        persona.scratch.last_meal_consumed = _now
      # A meal eaten AT the cafe is bought from the cafe counter, not taken
      # from the diner's home fridge (home fridges made every breakfast at
      # Hobbs Cafe a free meal: 18 sales/day, the owner's till never reached
      # her float and she earned nothing).
      if (keyword in ("eating", "breakfast", "lunch", "dinner", "meal")
          and os.environ.get("GA_HOME_FRIDGES", "1") == "1"
          and "hobbs cafe" in (persona.scratch.act_address or "").lower()):
        # Any counter food is a meal: pastries at breakfast, sandwiches
        # otherwise, whichever is left when the first choice is sold out.
        _cn = next((v for k, v in self.resource_manager.world_state.items()
                    if k.lower().endswith("hobbs cafe:cafe:counter")), {})
        _order = (("pastries", "sandwiches") if keyword == "breakfast"
                  else ("sandwiches", "pastries"))
        _pick = next((i for i in _order if _cn.get(i, 0) >= 1), _order[0])
        consumptions = [("counter", _pick, 1)]
        if keyword == "breakfast":              # a cup of coffee with it
          consumptions.append(("coffee machine", "coffee_beans", 0.1))
      # A grocery trip is paid ONCE, by _process_shopping (which also carries
      # the goods home). This mapping used to charge 5 eggs/3 bread/2 milk
      # on top for goods that went nowhere (communityexp4: ~$9 lost per trip).
      if (any(c[1] == "eggs" and c[0] in ("shelves", "grocery store shelf")
              for c in consumptions)
          and self.resource_manager.home_fridge_of.get(persona_name)):
        continue
      # Cafe prep is stocked by the venue (the daily wholesale delivery is
      # already booked in the economy ledger). Drawing it from the owner's
      # HOME fridge as well emptied it in a few sub-tasks and left her
      # repeatedly unable to cook (74 depletion events/day in communityexp).
      if (keyword in ("preparing cafe", "preparing food")
          and "hobbs cafe" in (persona.scratch.act_address or "").lower()):
        try:
          from economy import community_enabled
          if community_enabled():
            continue
        except Exception:
          pass

      for resource_pattern, item, amount in consumptions:
        # Find matching resource address based on pattern and persona location
        consumed = False

        for address in self.resource_manager.world_state:
          address_lower = address.lower()

          # Check if resource pattern matches
          if resource_pattern.lower() not in address_lower:
            continue

          # Prefer persona's own resources (their apartment/home)
          persona_first = persona_name.split()[0].lower()
          is_home = (persona_first in address_lower
                     or address == self.resource_manager.home_fridge_of.get(persona_name))

          # Check location match
          location_match = False
          if curr_location:
            curr_loc_lower = curr_location.lower()
            # Same building = same SECTOR. This compared loc_parts[:2], whose
            # first part is the world name ("the ville") — present in every
            # address, so ANY shelf matched from anywhere (Oct 22 2026).
            loc_parts = curr_loc_lower.split(":")
            sector = loc_parts[1].strip() if len(loc_parts) > 1 else ""
            if sector and sector in address_lower:
              location_match = True

          # Prioritize home resources, then location matches
          if is_home or location_match:
            # Check for shared resource contention (shower, hot_water)
            resource_type = None
            if "hot_water" in address_lower or "shower" in address_lower:
              resource_type = "hot_water"
            elif "bathroom" in address_lower:
              resource_type = "bathroom"

            if resource_type:
              success, wait_ticks, locked_by = self.resource_manager.try_acquire(
                address, resource_type, persona_name, self.step
              )
              if not success:
                # Resource is occupied - inject perception event
                self._inject_resource_occupied_event(persona, address, resource_type, locked_by, wait_ticks)
                all_success = False
                consumed = True  # Mark as handled
                break

            # Phase 5: Use purchase() for commercial locations (café counter, store)
            # Use plain consume() for home/personal resources
            is_commercial = (
              "counter" in address.lower() or
              "cafe" in address.lower() or
              "supply store" in address.lower() or
              "rose and crown pub" in address.lower() or
              "willows market" in address.lower() or
              ("shelves" in address.lower() and "store" in address.lower())
            )
            # Bug 2: Owner using their own café stock is COGS, not a sale
            CAFE_OWNER = "Isabella Rodriguez"
            is_owner_at_own_cafe = (
              persona.name == CAFE_OWNER and
              "hobbs cafe" in (persona.scratch.act_address or "").lower()
            )
            # STORE-OWNER GUARD (Sep 2026, post-Harvey): Carmen Ortiz now
            # owns the supply store — her consuming her own stock is COGS,
            # not a sale, same rule as Isabella's café.
            STORE_OWNER = "Carmen Ortiz"
            is_owner_at_own_store = (
              persona.name == STORE_OWNER and
              ("supply store" in (persona.scratch.act_address or "").lower()
               or "harvey oak supply store" in (persona.scratch.act_address or "").lower())
            )
            # Economy v2 (Oct 2026): same COGS guard for the pub and
            # market owners consuming their own venue's stock.
            OWN_VENUE_GUARDS = [
              ("Arthur Burton", "rose and crown pub"),
              ("John Lin", "willows market"),
            ]
            is_owner_at_own_venue = any(
              persona.name == _on and _vk in (persona.scratch.act_address or "").lower()
              for _on, _vk in OWN_VENUE_GUARDS)
            if is_commercial and not is_owner_at_own_cafe \
                and not is_owner_at_own_store \
                and not is_owner_at_own_venue \
                and hasattr(self.resource_manager, 'purchase'):
              success = self.resource_manager.purchase(address, item, amount, persona.scratch)
              if success:
                consumed = True
                # Economy v2 (Oct 2026): venue-routed sale credit — every
                # commercial purchase feeds the OWNING venue's till (weekly
                # owner draw replaces direct per-sale wallet credits).
                if self.economy is not None:
                  try:
                    _vid = self.economy.venue_of_address(address)
                    if _vid is not None:
                      _price = self.economy.item_price(item) * amount
                      self.economy.credit_sale(
                        _vid, item, amount, persona.name, _price)
                  except Exception as _econ_err:
                    print(f"[Economy] sale credit failed "
                          f"({_econ_err.__class__.__name__})")
                self._on_luxury_purchase(persona, item)
                try:  # a successful purchase resolves the stock-out lessons
                  from persona.cognitive_modules import lessons as _lessons
                  _lessons.clear(persona.scratch, "home_stock", item)
                  _lessons.clear(persona.scratch, "store_stock", item)
                except Exception:
                  pass
                # CAFÉ-ONLY credit (Sep 2026 fix): store sales already
                # credit the till inside purchase(); crediting Isabella here
                # too paid her twice — once with the store's own money.
                # Economy v2 (Oct 2026): owners are paid by weekly till
                # draw instead — this per-sale credit is superseded when
                # the v2 economy is active.
                addr_l = address.lower()
                if self.economy is None and ("cafe" in addr_l or "hobbs" in addr_l):
                  self._credit_cafe_sale(item, amount, persona.name)
                break
              else:
                _why = "sold_out"
                try:
                  if self.economy is not None and (
                      getattr(persona.scratch, "wallet", 0)
                      < self.economy.item_price(item) * amount):
                    _why = "cant_afford"
                except Exception:
                  pass
                self._inject_resource_depleted_event(persona, address, item, reason=_why)
                all_success = False
                consumed = True
                break
            else:
              success = self.resource_manager.consume(address, item, amount)
              if success:
                consumed = True
                break
              else:
                # Resource depleted - inject event
                self._inject_resource_depleted_event(persona, address, item)
                all_success = False
                consumed = True  # Mark as handled even though failed
                break

        # If no specific location matched, try any matching resource
        if not consumed:
          for address in self.resource_manager.world_state:
            if resource_pattern.lower() in address.lower():
              # never draw another household's fridge
              if "refrigerator" in address.lower() and (
                  address != self.resource_manager.home_fridge_of.get(persona_name)
                  and persona_first not in address.lower()):
                continue
              success = self.resource_manager.consume(address, item, amount)
              if success:
                consumed = True
                break

    # Handle production (restock cafe counters) — fires on regex match,
    # independent of the ACTION_RESOURCE_MAPPINGS keyword gate that
    # previously hid it (production fired 0 times in all n-series runs).
    # NOTE: hoisted OUT of the keyword loop — it must fire even when the
    # action matches no consumption keyword at all.
    prod_match = match_production(act_lower)
    if prod_match:
      prod_sw, prod_pa = prod_match
      act_addr = (getattr(persona.scratch, "act_address", "") or "").lower()
      curr_tile_addr = (getattr(persona.scratch, "curr_tile", "") or "")
      at_cafe = ("hobbs cafe" in act_addr
                 or "hobbs cafe" in str(curr_tile_addr).lower())
      if not at_cafe:
        # Also check description for cafe context
        at_cafe = ("hobbs cafe" in act_lower or "open cafe" in act_lower
                   or "cafe counter" in act_lower)
      if at_cafe:
        for address in self.resource_manager.world_state:
          if "counter" in address.lower() or "prepared_food" in address.lower():
            self.resource_manager.restock(address, "sandwiches", prod_sw)
            self.resource_manager.restock(address, "pastries", prod_pa)
        print(f"[ResourceManager] {persona.name} produced {prod_sw} sandwiches "
              f"and {prod_pa} pastries at the cafe counters")
        # Track accumulation so Isabella's stock-ups yield real artifacts
        self._note_cafe_production(persona, act_description)
      else:
        print(f"[ResourceManager] Production pattern matched for "
              f"{persona.name} but they are not at the cafe "
              f"(addr: '{act_addr}') — skipping restock")

    return all_success

  def _note_cafe_production(self, persona, act_description):
    """ECONOMY+ARTIFACT WELD: count Isabella's cafe stock-up actions; each
    full batch triggers the existing meal-artifact chain with a
    canonicalized description so the CREATION_ACTION_PATTERNS meal rules
    fire and a named, perceivable meal artifact lands on the cafe tile."""
    try:
      s = persona.scratch
      day_key = self.curr_time.strftime("%Y-%m-%d")
      key = f"cafe_production|{day_key}"
      if not hasattr(s, 'cafe_production_count'):
        s.cafe_production_count = {}
      s.cafe_production_count[key] = s.cafe_production_count.get(key, 0) + 1
      n = s.cafe_production_count[key]
      print(f"[Economy] {persona.name} cafe production count for {day_key}: {n}")
      # One batch-artifact per sim-day per creator — the stock-up block
      # fires many subtasks per day (measured ~86 in n32), and minting an
      # artifact every 3 fires would spam ~28 identical meals/day.
      done_key = f"cafe_artifact_done|{day_key}"
      if n >= 3 and not s.cafe_production_count.get(done_key):
        s.cafe_production_count[key] = 0
        s.cafe_production_count[done_key] = True
        # Canonicalized description so the meal creation rules match cleanly
        # (must contain a meal word — 'meal', not 'food' — or
        # match_creation_action returns None; verified against the real
        # CREATION_ACTION_PATTERNS matcher offline).
        canonical = "preparing a meal for the cafe: sandwiches and pastries batch"
        self.try_create_artifact(persona, canonical)
        print(f"[Economy] Cafe food batch complete for {persona.name} — "
              f"meal artifact attempted")
    except Exception as e:
      print(f"[Economy] cafe production tracking failed: {e}")

  def _credit_cafe_sale(self, item, amount, buyer_name="unknown"):
    """Phase 5: Credit Isabella Rodriguez's wallet when café items are purchased."""
    try:
      cafe_owner = "Isabella Rodriguez"
      if cafe_owner in self.personas:
        from resource_manager import ITEM_PRICES
        price = ITEM_PRICES.get(item, 1.0) * amount
        self.personas[cafe_owner].scratch.wallet = getattr(
          self.personas[cafe_owner].scratch, 'wallet', 250.0) + price
        # Update financial stress
        if hasattr(self.resource_manager, '_update_financial_stress'):
          self.resource_manager._update_financial_stress(
            self.personas[cafe_owner].scratch)
        print(f"[Economy] Isabella earned ${price:.2f} from {buyer_name} buying {amount}x {item} (wallet: ${self.personas[cafe_owner].scratch.wallet:.0f})")
    except Exception as e:
      pass  # Economy is optional, never crash sim

  def _sweep_store_till(self):
    """STORE-OWNER PAYDAY: weekly profit draw from the till (float $500
    kept). Delegates to resource_manager.sweep_store_till; returns 0 on
    any failure (economy must never crash the sim)."""
    try:
      if hasattr(self.resource_manager, 'sweep_store_till'):
        return self.resource_manager.sweep_store_till()
      return 0.0
    except Exception as e:
      print(f"[Payday] Warning: store till sweep failed: {e}")
      return 0.0

  def _perceive_venue_offer(self):
    """An agent standing in a shop sees what is on offer (stock + prices), once
    per venue per day, as a memory event — so "breakfast at Hobbs Cafe" can be
    chosen from what the counter really has."""
    if self.economy is None:
      return
    ws = self.resource_manager.world_state
    for name, persona in self.personas.items():
      try:
        s = persona.scratch
        vid = self.economy.venue_of_address(s.act_address or "")
        if vid is None:
          continue
        day = self.curr_time.toordinal()
        seen = s.__dict__.setdefault("_offer_seen", set())
        if (vid, day) in seen:
          continue
        seen.add((vid, day))
        from economy import VENUES
        key = VENUES[vid]["address_key"]
        have, out = [], []
        for addr, st in ws.items():
          if key not in addr.lower() or not addr.lower().startswith("the ville:"):
            continue
          for item, qty in st.items():
            if item in ("max", "last_delivery", "last_restocked") or not isinstance(qty, (int, float)):
              continue
            if item == "coffee_beans":          # sold by the cup, not by the bean
              if qty >= 1:
                have.append(f"fresh coffee (${self.economy.item_price('coffee'):.2f} a cup)")
              else:
                out.append("coffee")
              continue
            price = self.economy.item_price(item)
            if qty >= 1:
              have.append(f"{int(qty)} {item.replace('_', ' ')} (${price:.2f} each)")
            else:
              out.append(item.replace("_", " "))
        if not have and not out:
          continue
        desc = (f"{persona.name} looks over what {VENUES[vid]['name']} has on offer: "
                + (", ".join(have) if have else "nothing in stock")
                + (f". Sold out: {', '.join(out)}." if out else "."))
        from persona.prompt_template.gpt_structure import get_embedding
        try:
          emb = get_embedding(desc)
        except Exception:
          emb = [0.0] * 768
        persona.a_mem.add_event(
          self.curr_time, None, persona.name, "looks over", "menu", desc,
          {persona.name, VENUES[vid]["name"], "menu", "food"}, 3, (desc, emb), [])
      except Exception as e:
        print(f"[Reverie] venue offer failed for {name}: {e.__class__.__name__}")

  def _write_agent_cards(self):
    """Snapshot of each agent's live money/needs/fridge for the home-page
    cards (scratch.json only refreshes at checkpoints). Atomic write; the
    frontend reads resources/agent_cards.json."""
    try:
      civic = {}
      if self.economy is not None:
        civic = self.economy.state.get("civic", {}) or {}
      cards = {}
      for name, p in self.personas.items():
        sc = p.scratch
        first = name.split()[0].lower()
        fridge = {}
        _fa = self.resource_manager.home_fridge_of.get(name)
        if _fa is None:
          _fa = next((a for a in self.resource_manager.world_state
                      if "refrigerator" in a.lower() and first in a.lower()), None)
        if _fa in self.resource_manager.world_state:
          fridge = {k: round(v, 1) for k, v in self.resource_manager.world_state[_fa].items()
                    if isinstance(v, (int, float))}
        job = getattr(sc, "job", None)
        cards[name] = {
          "wallet": round(getattr(sc, "wallet", 0.0) or 0.0, 2),
          "stress": round(getattr(sc, "financial_stress", 0.0) or 0.0, 2),
          "needs": {k: round(v) for k, v in (getattr(sc, "needs", None) or {}).items()
                    if isinstance(v, (int, float))},
          "job": (job.get("venue") or job.get("role")) if isinstance(job, dict) else None,
          "civic": civic.get(name),
          "fridge": fridge,
          "goals": list(getattr(sc, "resource_goals", []) or [])[:3],
          "choice": getattr(sc, "choice_stats", None),
        }
      out = os.path.join(self.sim_folder, "resources", "agent_cards.json")
      tmp = out + ".tmp"
      with open(tmp, "w") as f:
        json.dump(cards, f)
      os.replace(tmp, out)
    except Exception as e:
      print(f"[Reverie] agent cards snapshot failed ({e.__class__.__name__})")

  def _check_and_do_payday(self):
    """Phase 5: Weekly payday — credit each agent a role-based income every
    7 sim-days. ON by default (fund-paid dividend); env
    GA_PAYDAY=0 disables. The store-owner till sweep is part of payday,
    so it is off with it; store income still flows through normal sales."""
    if os.environ.get("GA_PAYDAY", "1") != "1":
      return
    if os.environ.get("GA_COMMUNITY", "1") == "1":
      # Fund-paid dividend replaces the printed role-based incomes.
      if self.economy is not None:
        self.economy.pay_dividend(self.curr_time)
      return
    WEEKLY_INCOME = {
      "Isabella Rodriguez": 400.0,
      "Klaus Mueller": 200.0,
      "Maria Lopez": 150.0,
      "John Lin": 350.0,
      "Eddy Lin": 100.0,
      "Tom Moreno": 300.0,
      "Jane Moreno": 280.0,
      "Sam Moore": 180.0,
      "Giorgio Rossi": 260.0,
      "Ayesha Khan": 220.0,
    }
    # Trigger on Mondays at 6am sim time (every 7 days)
    is_monday = self.curr_time.weekday() == 0
    is_morning = self.curr_time.hour == 6 and self.curr_time.minute == 0
    last_payday = getattr(self, '_last_payday_day', -1)
    current_day = self.curr_time.toordinal()
    if is_monday and is_morning and current_day != last_payday:
      self._last_payday_day = current_day
      for persona_name, persona in self.personas.items():
        # STORE-OWNER SWEEP (Sep 2026, post-Harvey): Carmen Ortiz draws the
        # store till's weekly profit as her income — the till keeps a float
        # and she takes the surplus above $500. No profit, no payday; the
        # store is a real going concern whose income is customer-funded.
        STORE_OWNER = "Carmen Ortiz"
        if persona_name == STORE_OWNER:
          sweep = self._sweep_store_till()
          if sweep > 0:
            persona.scratch.wallet += sweep
            if hasattr(self.resource_manager, '_update_financial_stress'):
              self.resource_manager._update_financial_stress(persona.scratch)
            print(f"[Payday] {persona_name} swept ${sweep:.2f} store profit "
                  f"(wallet: ${persona.scratch.wallet:.0f})")
          else:
            print(f"[Payday] {persona_name} store till under float — no draw "
                  f"(wallet: ${persona.scratch.wallet:.0f})")
          continue
        income = WEEKLY_INCOME.get(persona_name, 150.0)
        persona.scratch.wallet = getattr(persona.scratch, 'wallet', 100.0) + income
        if hasattr(self.resource_manager, '_update_financial_stress'):
          self.resource_manager._update_financial_stress(persona.scratch)
        print(f"[Payday] {persona_name} received ${income:.0f} (wallet: ${persona.scratch.wallet:.0f})")

  def _inject_resource_depleted_event(self, persona, address, item, reason=None):
    """
    Inject a perception event into persona's memory when a resource is depleted.
    This triggers awareness and potential replanning.
    """
    try:
      # Parse the resource location for a human-readable description
      parts = address.split(":")
      location_name = parts[-1] if parts else address

      # Outcome memory (GA_LESSONS): remember WHY this failed so planning
      # stops repeating it (broke vs sold out vs own fridge empty).
      try:
        from persona.cognitive_modules import lessons as _lessons
        if _lessons.enabled():
          _al = address.lower()
          _commercial = any(k in _al for k in (
            "counter", "cafe", "supply store", "willows market",
            "rose and crown pub")) or ("shelves" in _al and "store" in _al)
          _lessons.record(persona.scratch,
                          "store_stock" if _commercial else "home_stock",
                          item, reason or "sold_out", self.curr_time)
      except Exception as _le:
        print(f"[Lessons] record failed ({_le.__class__.__name__})")

      # Create richer event description with alternatives based on item type
      food_items = ["eggs", "bread", "milk"]
      if any(food in item for food in food_items):
        event_desc = f"{persona.name} found the refrigerator empty (no {item}). They cannot prepare food at home. Hobbs Café serves breakfast nearby."
        poignancy = 8  # Higher importance for food depletion
      elif item == "coffee_beans":
        event_desc = f"{persona.name} found no coffee at home. Hobbs Café is nearby and serves coffee."
        poignancy = 8  # Higher importance for coffee depletion
      elif item == "hot_water":
        event_desc = f"{persona.name} found no hot water for a shower."
        poignancy = 5  # Moderate importance for hot water
      else:
        event_desc = f"The {location_name} is out of {item}"
        poignancy = 5  # Default importance

      # Add event to persona's associative memory
      curr_time = self.curr_time
      expiration = None

      s = location_name
      p = "is"
      o = f"out of {item}"

      keywords = {location_name.lower(), item.lower(), "empty", "depleted"}

      # Generate a simple embedding key
      embedding_key = f"resource_depleted_{persona.name}_{item}_{curr_time.strftime('%Y%m%d%H%M%S')}"

      # Create embedding pair (simple zero vector as placeholder)
      embedding_pair = (embedding_key, [0.0] * 1536)

      # Add the event to memory
      persona.a_mem.add_event(
        curr_time, expiration,
        s, p, o,
        event_desc, keywords, poignancy,
        embedding_pair, []
      )

      # Force replan by clearing act_address (causes act_check_finished() to return True)
      persona.scratch.act_address = None

      # Push a resource goal so replanning has direction (dedup — max 1 of each goal)
      if not hasattr(persona.scratch, "resource_goals"):
        persona.scratch.resource_goals = []
      if any(food in item for food in ["eggs", "bread", "milk"]):
        # FIX (shopping loop): eating out is a stopgap — also push a grocery
        # run so home fridges can actually restock. The store exists, is fully
        # stocked, and purchase() works; there was simply no path that ever
        # sent an agent to buy.
        goal = "having breakfast at Hobbs Cafe"
        if goal not in persona.scratch.resource_goals:
          persona.scratch.resource_goals.append(goal)
        self._queue_grocery_run(persona)
      elif item == "coffee_beans":
        goal = "getting coffee at Hobbs Cafe"
        if goal not in persona.scratch.resource_goals:
          persona.scratch.resource_goals.append(goal)
        self._queue_grocery_run(persona)
      elif item == "hot_water":
        goal = "use the common bathroom shower"
        if goal not in persona.scratch.resource_goals:
          persona.scratch.resource_goals.append(goal)
      else:
        goal = None
      if goal and goal not in persona.scratch.resource_goals:
        persona.scratch.resource_goals.append(goal)

      print(f"[ResourceManager] {persona.name} noticed: {event_desc}")

    except Exception as e:
      print(f"[ResourceManager] Error injecting depleted event: {e}")

  def _inject_resource_occupied_event(self, persona, address, resource_type, locked_by, wait_ticks):
    """
    Inject a perception event when a shared resource is occupied by another persona.
    """
    try:
      parts = address.split(":")
      location_name = parts[-2] if len(parts) >= 2 else parts[-1] if parts else address

      if resource_type == "hot_water" or resource_type == "shower":
        event_desc = f"The bathroom is occupied. {locked_by} is using the shower. {persona.name} will need to wait."
      elif resource_type == "bathroom":
        event_desc = f"The bathroom is occupied by {locked_by}. {persona.name} will need to wait."
      else:
        event_desc = f"The {location_name} is currently in use by {locked_by}."

      poignancy = 4  # Moderate importance

      curr_time = self.curr_time
      s = location_name
      p = "is"
      o = f"occupied by {locked_by}"
      keywords = {location_name.lower(), "occupied", "waiting", locked_by.lower().split()[0]}

      embedding_key = f"resource_occupied_{persona.name}_{resource_type}_{curr_time.strftime('%Y%m%d%H%M%S')}"
      embedding_pair = (embedding_key, [0.0] * 1536)

      persona.a_mem.add_event(
        curr_time, None,
        s, p, o,
        event_desc, keywords, poignancy,
        embedding_pair, []
      )

      print(f"[ResourceManager] {persona.name} found {resource_type} occupied by {locked_by}")

    except Exception as e:
      print(f"[ResourceManager] Error injecting occupied event: {e}")

  def satisfy_needs_for_action(self, persona, act_description):
    """
    Satisfy needs based on the action being performed.
    Called when a persona transitions to a new action.

    Mappings:
    - eat/breakfast/lunch/dinner/meal/food/snack/cook -> hunger +40, hydration +10
    - coffee/drink/water/tea/juice/beverage -> hydration +30, stimulation +10
    - shower/wash/bath/brush teeth -> hygiene +60
    - toilet/bathroom/restroom -> bladder +70
    - rest/nap/relax/sit/lounge -> comfort +30, energy +15
    - chat/talk/convers/meet/visit/party -> social +25
    - read/work/study/research/write/creat/paint/play -> stimulation +20
    """
    s = persona.scratch

    # Guard for backwards compatibility
    if not hasattr(s, "needs"):
      return

    if not act_description:
      return

    act_lower = act_description.lower()

    # Define keyword mappings
    satisfactions = []

    # Eating activities
    if any(kw in act_lower for kw in ["eat", "breakfast", "lunch", "dinner", "meal", "food", "snack", "cook"]):
      satisfactions.append(("hunger", 40))
      satisfactions.append(("hydration", 10))

    # Drinking activities
    if any(kw in act_lower for kw in ["coffee", "drink", "water", "tea", "juice", "beverage"]):
      satisfactions.append(("hydration", 30))
      satisfactions.append(("stimulation", 10))

    # Hygiene activities
    if any(kw in act_lower for kw in ["shower", "wash", "bath", "brush teeth"]):
      satisfactions.append(("hygiene", 60))

    # Bathroom activities
    if any(kw in act_lower for kw in ["toilet", "bathroom", "restroom"]):
      satisfactions.append(("bladder", 70))

    # Rest activities
    if any(kw in act_lower for kw in ["rest", "nap", "relax", "sit", "lounge"]):
      satisfactions.append(("comfort", 30))
      satisfactions.append(("energy", 15))

    # Social activities
    if any(kw in act_lower for kw in ["chat", "talk", "convers", "meet", "visit", "party"]):
      satisfactions.append(("social", 25))

    # Stimulating activities
    if any(kw in act_lower for kw in ["read", "work", "study", "research", "write", "creat", "paint", "play"]):
      satisfactions.append(("stimulation", 20))

    # Apply satisfactions
    for need, amount in satisfactions:
      if need in s.needs:
        s.needs[need] = min(100, s.needs[need] + amount)


  def start_server(self, int_counter):
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each personas to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break

      # <curr_env_file> file is the file that our frontend outputs. When the
      # frontend has done its job and moved the personas, then it will put a 
      # new environment file that matches our step count. That's when we run 
      # the content of this for loop. Otherwise, we just wait. 
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our personas. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
      
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked.
          # FIX (vector 1): pool the independent per-persona cognition with
          # a small thread pool; conversations serialize on _CONVO_LOCK.
          movements = {"persona": dict(),
                       "meta": dict()}

          # Async conversations: speakers whose chat is being generated on a
          # background thread are frozen for this step (snapshot, so a chat
          # finishing mid-step can't flip a persona half-way through).
          from concurrency_utils import ASYNC_FROZEN, start_pending_chats
          _frozen_now = dict(ASYNC_FROZEN)

          def _move_one(args):
            persona_name, persona = args
            # Layer 1: attribute all LLM calls on this worker thread to this
            # persona (per-persona temperature/think from town_model_settings)
            set_persona_call_context(persona_name)
            # Skip deceased personas
            if persona.scratch.is_deceased:
                return None
            # Frozen speaker: stand still with a chat bubble; the background
            # chat thread owns this persona until it writes the chat back.
            if persona_name in _frozen_now:
                return (persona_name, persona, persona.scratch.act_description,
                        self.personas_tile[persona_name], "💬",
                        f"conversing with {_frozen_now[persona_name]} @ "
                        f"{persona.scratch.act_address}")
            # Death check: has this agent reached their lifespan?
            if (persona.scratch.death_age is not None
                and persona.scratch.current_age >= persona.scratch.death_age):
                return persona_name  # handled serially below
            # Track previous action to detect transitions
            prev_act_description = persona.scratch.act_description
            # Fail-open guard (Oct 6 2026): ONE agent's LLM failure (timeout
            # past all retries, provider blip) must never wedge the whole
            # town — ex.map propagates any worker exception to reverie's
            # catch-all, which dumps the backend to the menu. Instead: log,
            # and reuse this agent's PREVIOUS move result so the step still
            # completes; the agent's plan retries naturally next tick.
            try:
              next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name],
              self.curr_time)
            except Exception as _mv_err:
              import traceback as _tb
              _tb.print_exc()
              print(f"[MoveFail] {persona_name}: {_mv_err.__class__.__name__} "
                    f"— agent skips this tick, town continues")
              return None
            return (persona_name, persona, prev_act_description,
                    next_tile, pronunciatio, description)

          dead_candidates = []
          with ThreadPoolExecutor(max_workers=GA_MOVE_WORKERS) as ex:
            move_results = list(ex.map(_move_one,
                                        list(self.personas.items())))
          # Serial point: start conversations queued during this step.
          start_pending_chats(set_persona_call_context)
          for res in move_results:
            if res is None:
              continue
            if isinstance(res, str):
              dead_candidates.append(res)
              continue
            persona_name, persona, prev_act_description, \
                next_tile, pronunciatio, description = res

            # FIX (vector 1): death handling moved out of the pooled worker —
            # milestone saves + tile removal are serial side effects.
            if persona_name in dead_candidates:
              persona.scratch.is_deceased = True
              curr_tile = self.personas_tile[persona_name]
              self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
              del self.personas_tile[persona_name]
              if self.bulletin_tile:
                bx, by = self.bulletin_tile
                death_notice = (
                  "community bulletin board", "reports",
                  f"{persona.name} passed away at age {persona.scratch.current_age}",
                  f"{persona.name} passed away at age {persona.scratch.current_age}")
                self.maze.tiles[by][bx]["events"].add(death_notice)
              chronicle_milestone(persona, "death",
                f"{persona.name} passed away at age {persona.scratch.current_age}",
                sim_folder, poignancy=9)
              persona.save()
              print(f"[Reverie] {persona.name} has passed away at age "
                    f"{persona.scratch.current_age}")
              continue

            # Birthday check on day boundary
            new_age = persona.scratch.current_age
            if new_age != persona.scratch.age:
              persona.scratch.age = new_age
              chronicle_milestone(persona, "birthday",
                f"{persona.name} turned {new_age}", sim_folder)

            # A purchase/consumption deferred until arrival fires on the step the agent gets there
            _pend = getattr(persona.scratch, "_pending_act", None)
            if _pend:
              if _pend != persona.scratch.act_description:
                persona.scratch._pending_act = None          # the plan changed first
              elif persona_name not in _frozen_now and self._arrived(persona):
                persona.scratch._pending_act = None
                try:
                  self.consume_resources_for_action(persona, _pend)
                  self._process_shopping(persona, _pend)
                  self._read_board_hook(persona, _pend)
                except Exception as e:
                  print(f"[Reverie] Warning: deferred consumption failed: {e}")

            # Satisfy needs when persona transitions to a new action
            curr_act_description = persona.scratch.act_description
            if persona_name in _frozen_now:
              curr_act_description = prev_act_description
            if curr_act_description != prev_act_description:
              self.satisfy_needs_for_action(persona, curr_act_description)
              # Consume world resources / shop only once the agent is THERE; until
              # then the action is pending (see the arrival check below).
              if self._arrived(persona):
                persona.scratch._pending_act = None
                try:
                  self.consume_resources_for_action(persona, curr_act_description)
                except Exception as e:
                  print(f"[Reverie] Warning: resource consumption failed: {e}")
                try:
                  self._process_shopping(persona, curr_act_description)
                except Exception as e:
                  print(f"[Reverie] Warning: shopping processing failed: {e}")
                self._read_board_hook(persona, curr_act_description)
              else:
                persona.scratch._pending_act = curr_act_description
              # Artifact creation and interaction (Phase 6)
              try:
                self.try_create_artifact(persona, curr_act_description)
                self.process_artifact_interaction(persona, curr_act_description)
              except Exception as e:
                print(f"[Reverie] Warning: artifact processing failed: {e}")
              # Ville Dots: simulated-PC work session on matching actions
              try:
                if matches_pc_session(curr_act_description):
                  self._run_pc_session_for(persona, curr_act_description)
              except Exception as e:
                print(f"[VilleOS] Warning: PC session hook failed: {e}")
              # Ville Dots: ingest external feedback notes into memory
              try:
                self._drain_pc_notes_for(persona)
              except Exception as e:
                print(f"[VilleOS] Warning: note ingestion failed: {e}")

            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Add artifact data for frontend rendering
          try:
            artifact_data = {}
            for art in self.artifact_manager.get_all_active_artifacts():
              location = art["location"]
              if location in self.maze.address_tiles:
                tile = list(self.maze.address_tiles[location])[0]
                emoji = ARTIFACT_EMOJI.get(art["type"], "📦")
                artifact_data[art["id"]] = {
                  "x": tile[0], "y": tile[1],
                  "emoji": emoji,
                  "label": art["name"],
                  "creator": art["creator"],
                  "type": art["type"],
                }
            movements["artifacts"] = artifact_data
          except Exception as e:
            movements["artifacts"] = {}

          # Include the meta information about the current stage in the
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          os.makedirs(f"{sim_folder}/movement", exist_ok=True)
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))

          # After this cycle, the world takes one step forward, and the 
          # current time moves by <sec_per_step> amount. 
          # Conversations do NOT move the world clock (Oct 22 2026): they
          # take the participants' time — each reply lengthens the chat
          # activity (plan.generate_convo / chat_time.conversation_minutes)
          # and the pair's turns pass in conversation like any activity.
          _chat_debt, _chat_utts = _chat_time.drain_tick()
          if _chat_utts:
            print(f"[ChatTime] step {self.step}: {_chat_utts} replies "
                  f"generated (participants' time, not the world clock)")
          self.step += 1
          _step_s = self._night_step_seconds()
          self.curr_time += datetime.timedelta(seconds=_step_s)

          # Keep curr_step.json up to date so the frontend can re-initialize
          # correctly on page refresh (instead of always starting from step 0).
          curr_step = {"step": self.step}
          with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile: 
            outfile.write(json.dumps(curr_step, indent=2))

          # Periodic checkpoint every 360 steps (1 sim-hour): persist
          # persona memories + meta so a crashed run can be resumed by
          # forking from this sim without losing more than an hour of state.
          if self.step % 360 == 0:
            try:
              self.save()
              print(f"[Reverie] Checkpoint saved at step {self.step}")
            except Exception as e:
              print(f"[Reverie] Warning: periodic checkpoint failed: {e}")
            # AGENCY L2: refresh per-persona routine libraries from real
            # movement history (LLM-free mining, atomic writes).
            try:
              import routines as _rt
              _n = _rt.write_routines()
              if _n:
                print(f"[Routines] harvested for {_n} personas at step {self.step}")
            except Exception as _rt_err:
              print(f"[Routines] harvest failed: {_rt_err}")

          int_counter -= 1

          # Tick agent needs each step
          self.tick_needs(_step_s / float(self.sec_per_step))
          # Status need: lingering thoughts, never an emergency replan
          self._status_thoughts()

          # Tick world resources (hot water refill, daily deliveries)
          try:
            self.resource_manager.tick(self.curr_time)
          except Exception as e:
            print(f"[Reverie] Warning: resource_manager.tick failed: {e}")

          # Economy v2 (Oct 2026): calendar flows + hiring encounters.
          # All paths fail-open internally; the guard here is belt+braces.
          if self.economy is not None:
            try:
              self.economy.pay_restock_worker(self.curr_time)
              self.economy.weekly_flows(self.curr_time)
              self.economy.consider_hiring(self.curr_time)
              self.economy.record_presence(self.curr_time)
              self.economy.pay_employees(self.curr_time)
              # hiring encounters: co-located applicant+owner
              _paths = {}
              for _pn, _p in self.personas.items():
                try:
                  _paths[_pn] = self.maze.get_tile_path(
                    _p.scratch.curr_tile, "arena")
                except Exception:
                  pass
              self.economy.try_hire_encounters(_paths)
            except Exception as _econ_err:
              print(f"[Economy] tick failed ({_econ_err.__class__.__name__})")

          if self.step % 6 == 0:
            self._write_agent_cards()
            self._perceive_venue_offer()

          # Phase 5: Weekly payday check
          try:
            self._check_and_do_payday()
          except Exception as e:
            pass  # Economy is optional, never crash sim

      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)


  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    while True: 
      sim_command = input("Enter option: ")
      sim_command = sim_command.strip()
      ret_str = ""

      try: 
        if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
          # Finishes the simulation environment and saves the progress. 
          # Example: fin
          self.save()
          break

        elif sim_command.lower() == "start path tester mode": 
          # Starts the path tester and removes the currently forked sim files.
          # Note that once you start this mode, you need to exit out of the
          # session and restart in case you want to run something else. 
          shutil.rmtree(sim_folder) 
          self.start_path_tester_server()

        elif sim_command.lower() == "exit": 
          # Finishes the simulation environment but does not save the progress
          # and erases all saved data from current simulation. 
          # Example: exit 
          shutil.rmtree(sim_folder) 
          break 

        elif sim_command.lower() == "save": 
          # Saves the current simulation progress. 
          # Example: save
          self.save()

        elif sim_command[:3].lower() == "run": 
          # Runs the number of steps specified in the prompt.
          # Example: run 1000
          int_count = int(sim_command.split()[-1])
          rs.start_server(int_count)

        elif ("print persona schedule" 
              in sim_command[:22].lower()): 
          # Print the decomposed schedule of the persona specified in the 
          # prompt.
          # Example: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_summary())

        elif ("print all persona schedule" 
              in sim_command[:26].lower()): 
          # Print the decomposed schedule of all personas in the world. 
          # Example: print all persona schedule
          for persona_name, persona in self.personas.items(): 
            ret_str += f"{persona_name}\n"
            ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
            ret_str += f"---\n"

        elif ("print hourly org persona schedule" 
              in sim_command.lower()): 
          # Print the hourly schedule of the persona specified in the prompt.
          # This one shows the original, non-decomposed version of the 
          # schedule.
          # Ex: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_hourly_org_summary())

        elif ("print persona current tile" 
              in sim_command[:26].lower()): 
          # Print the x y tile coordinate of the persona specified in the 
          # prompt. 
          # Ex: print persona current tile Isabella Rodriguez
          ret_str += str(self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.curr_tile)

        elif ("print persona chatting with buffer" 
              in sim_command.lower()): 
          # Print the chatting with buffer of the persona specified in the 
          # prompt.
          # Ex: print persona chatting with buffer Isabella Rodriguez
          curr_persona = self.personas[" ".join(sim_command.split()[-2:])]
          for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
            ret_str += f"{p_n}: {count}"

        elif ("print persona associative memory (event)" 
              in sim_command.lower()):
          # Print the associative memory (event) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (event) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_events())

        elif ("print persona associative memory (thought)" 
              in sim_command.lower()): 
          # Print the associative memory (thought) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (thought) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_thoughts())

        elif ("print persona associative memory (chat)" 
              in sim_command.lower()): 
          # Print the associative memory (chat) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (chat) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_chats())

        elif ("print persona spatial memory" 
              in sim_command.lower()): 
          # Print the spatial memory of the persona specified in the prompt
          # Ex: print persona spatial memory Isabella Rodriguez
          self.personas[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

        elif ("print current time" 
              in sim_command[:18].lower()): 
          # Print the current time of the world. 
          # Ex: print current time
          ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
          ret_str += f'steps: {self.step}'

        elif ("print tile event" 
              in sim_command[:16].lower()): 
          # Print the tile events in the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
          for i in self.maze.access_tile(cooordinate)["events"]: 
            ret_str += f"{i}\n"

        elif ("print tile details" 
              in sim_command.lower()): 
          # Print the tile details of the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
          for key, val in self.maze.access_tile(cooordinate).items(): 
            ret_str += f"{key}: {val}\n"

        elif ("call -- analysis" 
              in sim_command.lower()): 
          # Starts a stateless chat session with the agent. It does not save 
          # anything to the agent's memory. 
          # Ex: call -- analysis Isabella Rodriguez
          persona_name = sim_command[len("call -- analysis"):].strip() 
          self.personas[persona_name].open_convo_session("analysis")

        elif ("call -- load history" 
              in sim_command.lower()): 
          curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
          # call -- load history the_ville/agent_history_init_n3.csv

          rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
          clean_whispers = []
          for row in rows: 
            agent_name = row[0].strip() 
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers: 
              clean_whispers += [[agent_name, whisper]]

          load_history_via_whisper(self.personas, clean_whispers)

        print (ret_str)

      except:
        traceback.print_exc()
        print ("Error.")
        pass


if __name__ == '__main__':
  # rs = ReverieServer("base_the_ville_isabella_maria_klaus", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-1")
  # rs = ReverieServer("July1_the_ville_isabella_maria_klaus-step-3-20", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-21")
  # rs.open_server()

  origin = input("Enter the name of the forked simulation: ").strip()
  target = input("Enter the name of the new simulation: ").strip()

  rs = ReverieServer(origin, target)
  rs.open_server()




















































