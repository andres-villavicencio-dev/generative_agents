"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: associative_memory.py
Description: Defines the core long-term memory module for generative agents.

Note (May 1, 2023) -- this class is the Memory Stream module in the generative
agents paper. 
"""
import sys
sys.path.append('../../')

import json
import datetime
import os
import re

from global_methods import *


# ---------------------------------------------------------------------------
# THOUGHT DEDUP (GA_THOUGHT_DEDUP, default on, Nov 2026): the insight and
# reflection layers kept re-writing the same thought with cosmetic edits —
# 63% of the memory lines an agent carried into its chat prompts were
# near-duplicates (measured on run n76's utterance traces), inflating every
# utterance prompt. A new thought that is too similar to one already added in
# the last 24h is rejected at intake. The once-daily plan marker is exempt
# (it is near-identical every day by design). Fail-open: any error in the gate
# lets the thought through; the sim never breaks over a dedup.
# ---------------------------------------------------------------------------
THOUGHT_DEDUP_LOOKBACK_H = 24
THOUGHT_DEDUP_JACCARD = 0.45


def thought_dedup_enabled():
  """GA_THOUGHT_DEDUP, default ON (project rule: feature flags are on unless set off)."""
  return os.environ.get("GA_THOUGHT_DEDUP", "1") != "0"


def _dedup_words(text):
  return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2}


def _is_plan_marker(description):
  """True for the daily 'This is X's plan for <day>' thought: a deliberate
  once-a-day marker whose near-identical wording is not spam."""
  t = str(description or "")
  return t.startswith("This is ") and "'s plan for " in t


def _trace_dedup(created, who, new_desc, old_desc):
  """Append one suppression record to <GA_TRACE_DIR>/thought_dedup.jsonl (fail-open)."""
  try:
    base = os.environ.get("GA_TRACE_DIR") or os.path.join(
      os.path.dirname(__file__), "../../../ga_traces")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "thought_dedup.jsonl"), "a") as f:
      f.write(json.dumps({"t": str(created), "who": str(who)[:60],
                          "desc": str(new_desc)[:160],
                          "dup_of": str(old_desc)[:160]}) + "\n")
  except Exception:
    pass


def _thought_is_dup(mem, created, description):
  """True when <description> is too close to a thought from the last
  LOOKBACK hours. <mem.seq_thought> is newest-first, so the scan stops at
  the lookback cutoff. Returns False on any error (fail-open)."""
  if created is None or _is_plan_marker(description):
    return False
  try:
    new = _dedup_words(description)
    if not new:
      return False
    cutoff = created - datetime.timedelta(hours=THOUGHT_DEDUP_LOOKBACK_H)
    for node in mem.seq_thought:
      if node.created is None or node.created < cutoff:
        break
      if _is_plan_marker(node.description):
        continue
      old = _dedup_words(node.description)
      if old and len(new & old) / len(new | old) >= THOUGHT_DEDUP_JACCARD:
        _trace_dedup(created, node.subject, description, node.description)
        return True
  except Exception:
    return False
  return False


class ConceptNode: 
  def __init__(self,
               node_id, node_count, type_count, node_type, depth,
               created, expiration, 
               s, p, o, 
               description, embedding_key, poignancy, keywords, filling): 
    self.node_id = node_id
    self.node_count = node_count
    self.type_count = type_count
    self.type = node_type # thought / event / chat
    self.depth = depth

    self.created = created
    self.expiration = expiration
    self.last_accessed = self.created

    self.subject = s
    self.predicate = p
    self.object = o

    self.description = description
    self.embedding_key = embedding_key
    self.poignancy = poignancy
    self.keywords = keywords
    self.filling = filling


  def spo_summary(self): 
    return (self.subject, self.predicate, self.object)


class AssociativeMemory: 
  def __init__(self, f_saved): 
    self.id_to_node = dict()

    self.seq_event = []
    self.seq_thought = []
    self.seq_chat = []

    self.kw_to_event = dict()
    self.kw_to_thought = dict()
    self.kw_to_chat = dict()

    self.kw_strength_event = dict()
    self.kw_strength_thought = dict()

    self.embeddings = json.load(open(f_saved + "/embeddings.json"))

    nodes_load = json.load(open(f_saved + "/nodes.json"))
    for count in range(len(nodes_load.keys())): 
      node_id = f"node_{str(count+1)}"
      node_details = nodes_load[node_id]

      node_count = node_details["node_count"]
      type_count = node_details["type_count"]
      node_type = node_details["type"]
      depth = node_details["depth"]

      created = datetime.datetime.strptime(node_details["created"], 
                                           '%Y-%m-%d %H:%M:%S')
      expiration = None
      if node_details["expiration"]: 
        expiration = datetime.datetime.strptime(node_details["expiration"],
                                                '%Y-%m-%d %H:%M:%S')

      s = node_details["subject"]
      p = node_details["predicate"]
      o = node_details["object"]

      description = node_details["description"]
      embedding_pair = (node_details["embedding_key"], 
                        self.embeddings[node_details["embedding_key"]])
      poignancy =node_details["poignancy"]
      keywords = set(node_details["keywords"])
      filling = node_details["filling"]
      
      if node_type == "event": 
        self.add_event(created, expiration, s, p, o, 
                   description, keywords, poignancy, embedding_pair, filling)
      elif node_type == "chat": 
        self.add_chat(created, expiration, s, p, o, 
                   description, keywords, poignancy, embedding_pair, filling)
      elif node_type == "thought": 
        self.add_thought(created, expiration, s, p, o, 
                   description, keywords, poignancy, embedding_pair, filling)

    kw_strength_load = json.load(open(f_saved + "/kw_strength.json"))
    if kw_strength_load["kw_strength_event"]: 
      self.kw_strength_event = kw_strength_load["kw_strength_event"]
    if kw_strength_load["kw_strength_thought"]: 
      self.kw_strength_thought = kw_strength_load["kw_strength_thought"]

    
  def save(self, out_json): 
    r = dict()
    for count in range(len(self.id_to_node.keys()), 0, -1): 
      node_id = f"node_{str(count)}"
      node = self.id_to_node[node_id]

      r[node_id] = dict()
      r[node_id]["node_count"] = node.node_count
      r[node_id]["type_count"] = node.type_count
      r[node_id]["type"] = node.type
      r[node_id]["depth"] = node.depth

      r[node_id]["created"] = node.created.strftime('%Y-%m-%d %H:%M:%S')
      r[node_id]["expiration"] = None
      if node.expiration: 
        r[node_id]["expiration"] = (node.expiration
                                        .strftime('%Y-%m-%d %H:%M:%S'))

      r[node_id]["subject"] = node.subject
      r[node_id]["predicate"] = node.predicate
      r[node_id]["object"] = node.object

      r[node_id]["description"] = node.description
      r[node_id]["embedding_key"] = node.embedding_key
      r[node_id]["poignancy"] = node.poignancy
      r[node_id]["keywords"] = list(node.keywords)
      r[node_id]["filling"] = node.filling

    with open(out_json+"/nodes.json", "w") as outfile:
      json.dump(r, outfile)

    r = dict()
    r["kw_strength_event"] = self.kw_strength_event
    r["kw_strength_thought"] = self.kw_strength_thought
    with open(out_json+"/kw_strength.json", "w") as outfile:
      json.dump(r, outfile)

    with open(out_json+"/embeddings.json", "w") as outfile:
      json.dump(self.embeddings, outfile)


  def add_event(self, created, expiration, s, p, o, 
                      description, keywords, poignancy, 
                      embedding_pair, filling):
    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_event) + 1
    node_type = "event"
    node_id = f"node_{str(node_count)}"
    depth = 0

    # Node type specific clean up. 
    if "(" in description: 
      description = (" ".join(description.split()[:3]) 
                     + " " 
                     +  description.split("(")[-1][:-1])

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], 
                       poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_event[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_event: 
        self.kw_to_event[kw][0:0] = [node]
      else: 
        self.kw_to_event[kw] = [node]
    self.id_to_node[node_id] = node 

    # Adding in the kw_strength
    if f"{p} {o}" != "is idle":  
      for kw in keywords: 
        if kw in self.kw_strength_event: 
          self.kw_strength_event[kw] += 1
        else: 
          self.kw_strength_event[kw] = 1

    self.embeddings[embedding_pair[0]] = embedding_pair[1]

    return node


  def add_thought(self, created, expiration, s, p, o, 
                        description, keywords, poignancy, 
                        embedding_pair, filling):
    # THOUGHT DEDUP (GA_THOUGHT_DEDUP): reject a near-duplicate of a recent
    # thought at intake, so reflection spam never reaches the prompts. The
    # caller's embedding work is already done; we only skip storing the node.
    # Returns None (no caller uses the return value; fail-open on errors).
    if thought_dedup_enabled() and _thought_is_dup(self, created, description):
      print(f"[ThoughtDedup] {s}: suppressed duplicate thought: {str(description)[:70]}")
      return None

    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_thought) + 1
    node_type = "thought"
    node_id = f"node_{str(node_count)}"
    depth = 1 
    try: 
      if filling: 
        depth += max([self.id_to_node[i].depth for i in filling])
    except: 
      pass

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_thought[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_thought: 
        self.kw_to_thought[kw][0:0] = [node]
      else: 
        self.kw_to_thought[kw] = [node]
    self.id_to_node[node_id] = node 

    # Adding in the kw_strength
    if f"{p} {o}" != "is idle":  
      for kw in keywords: 
        if kw in self.kw_strength_thought: 
          self.kw_strength_thought[kw] += 1
        else: 
          self.kw_strength_thought[kw] = 1

    self.embeddings[embedding_pair[0]] = embedding_pair[1]

    return node


  def add_chat(self, created, expiration, s, p, o, 
                     description, keywords, poignancy, 
                     embedding_pair, filling): 
    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_chat) + 1
    node_type = "chat"
    node_id = f"node_{str(node_count)}"
    depth = 0

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_chat[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_chat: 
        self.kw_to_chat[kw][0:0] = [node]
      else: 
        self.kw_to_chat[kw] = [node]
    self.id_to_node[node_id] = node 

    self.embeddings[embedding_pair[0]] = embedding_pair[1]
        
    return node


  def get_summarized_latest_events(self, retention): 
    ret_set = set()
    for e_node in self.seq_event[:retention]: 
      ret_set.add(e_node.spo_summary())
    return ret_set


  def get_str_seq_events(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_event): 
      ret_str += f'{"Event", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n'
    return ret_str


  def get_str_seq_thoughts(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_thought): 
      ret_str += f'{"Thought", len(self.seq_thought) - count, ": ", event.spo_summary(), " -- ", event.description}'
    return ret_str


  def get_str_seq_chats(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_chat): 
      ret_str += f"with {event.object.content} ({event.description})\n"
      ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
      for row in event.filling: 
        ret_str += f"{row[0]}: {row[1]}\n"
    return ret_str


  def retrieve_relevant_thoughts(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]

    ret = []
    for i in contents: 
      if i in self.kw_to_thought: 
        ret += self.kw_to_thought[i.lower()]

    ret = set(ret)
    return ret


  def retrieve_relevant_events(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]

    ret = []
    for i in contents: 
      if i in self.kw_to_event: 
        ret += self.kw_to_event[i]

    ret = set(ret)
    return ret


  def get_last_chat(self, target_persona_name): 
    if target_persona_name.lower() in self.kw_to_chat: 
      return self.kw_to_chat[target_persona_name.lower()][0]
    else: 
      return False



























