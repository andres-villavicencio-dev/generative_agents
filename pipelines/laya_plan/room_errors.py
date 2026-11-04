import json, sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import persona.prompt_template.laya_scoring as L
L.LAYA_URL = "http://127.0.0.1:" + sys.argv[1]
from persona.prompt_template.laya_loc import ask_level
items = [json.loads(l) for l in open("/home/andus/Projects/laya-test/v7_data/test.jsonl")]
items = [i for i in items if i["kind"] == "loc" and i["state"]["level"] == "room"]
errs = []
for it in items:
    q = it["questions"]; opts = list(q["loc"]["criteria"])
    pr = ask_level(it["state"], q, opts)
    if not pr: continue
    pred = max(pr, key=pr.get); lab = opts[it["label"]]
    if pred != lab:
        p = sorted(pr.values(), reverse=True)
        errs.append({"persona": it["persona"], "ctx": it["state"]["situation"], "area": it["state"]["situation"].split("Chosen area: ")[-1], "crit": q["loc"]["criteria"], "label": lab, "pred": pred, "opts": opts, "margin": p[0] - p[1], "n_opts": len(opts), "label_conf": pr.get(lab)})
json.dump(errs, open("/tmp/room_errors.json", "w"))
print(len(errs), "errors of", len(items))
