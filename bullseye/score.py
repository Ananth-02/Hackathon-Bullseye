"""Step 4: the scoring flowchart from the team doc, as code.

Every decision is appended to `path`, so the viewer can light up the exact
route a function took through the flowchart.
"""
LEVELS = ["low", "medium", "high"]
TRIGGER_TEXT = {
    "isr_context": "runs in interrupt context",
    "writes_hw_register": "writes a hardware register",
    "critical_section": "critical section / interrupt masking",
    "busy_wait": "busy-wait or timing delay",
    "locked_context": "only ever runs inside a critical section / lock",
    "watchdog": "services or configures the watchdog",
    "external_interface": "external interface (public API)",
}


def score(node, evidence, patterns, llm, cfg):
    th = cfg.get("thresholds", {"high": 15, "medium": 5})
    hard = cfg.get("hard_triggers", list(TRIGGER_TEXT))
    path, reasons = [], []

    triggers = [t for t in hard if node["flags"].get(t)]
    if triggers:
        risk = "high"
        path.append("trigger:yes")
        reasons += [TRIGGER_TEXT[t] for t in triggers]
    else:
        path.append("trigger:no")
        fi = node["fan_in"]
        if fi >= th["high"]:
            risk, band = "high", "high"
        elif fi >= th["medium"]:
            risk, band = "medium", "medium"
        else:
            risk, band = "low", "low"
        path.append(f"fanin:{band}")
        reasons.append(f"{fi} direct caller{'s' if fi != 1 else ''} (blast radius {node['blast_radius']})")
        floors = [t for t in cfg.get("floor_triggers", []) if node["flags"].get(t)]
        if floors:
            path.append("api:yes")
            triggers = floors
            reasons += [TRIGGER_TEXT[t] + ": at least MEDIUM" for t in floors]
            if risk == "low":
                risk = "medium"
        else:
            path.append("api:no")

    if llm is None:
        path.append("llm:offline")
    elif str(llm.get("looks_deliberate", "")).lower() == "yes":
        path.append("llm:yes")
        if risk != "high":
            risk = LEVELS[LEVELS.index(risk) + 1]
            reasons.append("LLM: looks deliberate / defensive, risk raised one level")
    else:
        path.append("llm:no")

    commits = [e for e in evidence if e["type"] == "commit"]
    comments = [e for e in evidence if e["type"] == "comment"]
    if comments or commits:
        conf = "high"
        path.append("evidence:comment" if comments else "evidence:commit")
    elif patterns:
        conf = "medium"
        path.append("evidence:pattern")
    else:
        conf = "low"
        path.append("evidence:none")

    down = []
    if node["fn"].indirect_calls:
        down.append(f"{node['fn'].indirect_calls} call(s) through function pointers inside")
    if node["refs_in"]:
        down.append("also used as a callback / pointer, so some callers are invisible")
    if llm and llm.get("models_disagree"):
        down.append("the two LLMs disagree")
    if down:
        path.append("downgrade:yes")
        conf = LEVELS[max(0, LEVELS.index(conf) - 1)]
    else:
        path.append("downgrade:no")
    return risk, conf, path, reasons, down, triggers


def open_questions(node, triggers, conf, evidence):
    q = []
    n = node["notes"]
    if "writes_hw_register" in triggers and conf != "high":
        q.append("Which hardware behaviour do these register writes depend on? No comment or commit explains them.")
    if "busy_wait" in triggers:
        q.append("What condition does the wait loop depend on, and what happens if the hardware never sets it (is there a timeout)?")
    if "isr_context" in triggers:
        q.append("Is everything this function does safe inside an interrupt (no blocking, bounded time)?")
    if node["refs_in"]:
        q.append(f"It is passed as a pointer in {', '.join(r.split('::')[-1] for r in node['refs_in'][:3])}: who calls it at runtime?")
    if node["fn"].indirect_calls:
        q.append("Which functions can the pointer calls inside it reach?")
    if node["patterns"] and not any(e["type"] == "comment" for e in evidence):
        q.append(f"The code contains a known defensive pattern ({node['patterns'][0]}). Which failure was it added for?")
    return q[:3]
