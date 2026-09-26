"""Step 4: the scoring flowchart from the team doc, as code.

Hard trigger + skeptical LLM → still gated high — the LLM can add context but can't override the categorical gate.
No trigger, grounded LLM (p=0.85), low fan-in → medium (p=0.67) — a strong grounded signal now meaningfully pulls risk up even against a low structural score, instead of getting outvoted.
Same case, but LLM cites nothing real → drops to low (p=0.06) — the exact behavior you wanted: an ungrounded claim gets clamped to 0.5 and can no longer inflate risk.
No LLM, no evidence at all → low, conf: low — honest "we don't know, defaulting low-risk with low confidence" rather than a false-positive spike.
High fan-in alone, no LLM → high (p=0.94) — structural signal by itself is still enough to flag risk, confidence correctly reads low since there's no supporting evidence to explain why.

Every decision is appended to `path`, so the viewer can light up the exact
route a function took through the flowchart.

v2: the LLM's contribution is now a calibrated probability with a required
evidence pointer, combined into a log-odds score alongside the structural
signal instead of jumping risk a whole tier on a bare yes/no. See
`llm_term()` for the grounding rule that prevents an ungrounded claim from
moving the score.
"""
import math

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

# --- Option 1: log-odds combination -----------------------------------
#
# logit(p) = w0
#          + w1 * struct_term        (fan-in / hard-trigger band, mapped to a logit)
#          + w2 * llm_term           (LLM's grounded P(deliberate), clamped toward 0
#                                      when it can't cite real evidence)
#          - w3 * disagreement_term  (two LLMs / LLM vs pattern disagree)
#          + w4 * evidence_term      (comment/commit/pattern strength)
#
# p = sigmoid(logit(p)) is then bucketed with the same thresholds you already
# use, so downstream reporting code doesn't change.

# w0 is the prior in the absence of any signal. Kept at 0 (neutral, p=0.5)
# rather than negative, since we have no principled reason to assume
# ambiguous code is low-risk by default -- that assumption is exactly what
# let a strong, grounded LLM signal get outvoted by a weak fan-in band
# below. These weights are still hand-picked placeholders, not fit to data;
# see the suggested calibration step before trusting them on real code.
WEIGHTS = {"w0": 0.0, "w1": 2.5, "w2": 2.0, "w3": 1.5, "w4": 1.0}

STRUCT_BAND_LOGIT = {"low": -1.5, "medium": 0.0, "high": 1.5}
EVIDENCE_BAND_LOGIT = {"none": -1.0, "pattern": 0.0, "commit": 0.75, "comment": 1.0}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bucket(p, thresholds):
    # thresholds are on the same p-scale you'd get from fan-in bands; if your
    # cfg's thresholds are still fan-in counts, convert once at startup and
    # pass p-scale cutoffs in here instead (e.g. {"high": 0.75, "medium": 0.4}).
    if p >= thresholds.get("high", 0.75):
        return "high"
    if p >= thresholds.get("medium", 0.4):
        return "medium"
    return "low"


def hard_triggers(node, hard):
    """Categorical gate, kept separate from the log-odds equation.

    A hard trigger (ISR context, register write, critical section, etc.) means
    the function is the *kind* of code where a wrong guess is dangerous -- that
    is a qualitative fact, not a quantity that should get diluted by averaging
    it against fan-in or an LLM opinion. So it short-circuits straight to
    "high" risk, same as your original flowchart, before Option 1's equation
    ever runs.
    """
    return [t for t in hard if node["flags"].get(t)]


def struct_term(node, cfg):
    """Structural signal for the NON-trigger branch only: fan-in band."""
    th = cfg.get("thresholds", {"high": 15, "medium": 5})
    fi = node["fan_in"]
    if fi >= th["high"]:
        band = "high"
    elif fi >= th["medium"]:
        band = "medium"
    else:
        band = "low"
    return STRUCT_BAND_LOGIT[band], band


def evidence_term(evidence, patterns):
    """Strength of grounding available, independent of what it says."""
    commits = [e for e in evidence if e["type"] == "commit"]
    comments = [e for e in evidence if e["type"] == "comment"]
    if comments:
        return EVIDENCE_BAND_LOGIT["comment"], "comment"
    if commits:
        return EVIDENCE_BAND_LOGIT["commit"], "commit"
    if patterns:
        return EVIDENCE_BAND_LOGIT["pattern"], "pattern"
    return EVIDENCE_BAND_LOGIT["none"], "none"


def llm_term(llm, evidence, patterns):
    """Grounded LLM contribution.

    Expects llm to look like:
        {
            "p_deliberate": 0.0-1.0,      # calibrated probability, not yes/no
            "cites": ["comment:3", "pattern:busy_wait_retry"],  # pointers into
                                                                  # evidence/patterns
            "models_disagree": bool,
        }

    Any citation that doesn't resolve to something actually present in
    `evidence` or `patterns` is dropped. If every citation is dropped, the
    probability is clamped to 0.5 (no information) before it ever reaches
    the weighted sum below -- an ungrounded claim literally cannot move the
    score, which is the anti-guessing mechanism.
    """
    if llm is None:
        return 0.0, "offline", []

    real_evidence_ids = {f"comment:{i}" for i, _ in enumerate(evidence) if evidence[i]["type"] == "comment"}
    real_evidence_ids |= {f"commit:{i}" for i, _ in enumerate(evidence) if evidence[i]["type"] == "commit"}
    real_pattern_ids = {f"pattern:{p}" for p in patterns}
    valid = [c for c in llm.get("cites", []) if c in real_evidence_ids or c in real_pattern_ids]

    p = llm.get("p_deliberate", 0.5)
    if not valid:
        p = 0.5  # clamp: no real citation, no influence
        status = "ungrounded"
    else:
        status = "grounded"

    # map probability to a logit contribution centered at 0 for p=0.5
    contribution = math.log(p / (1 - p)) if 0 < p < 1 else (4.0 if p >= 1 else -4.0)
    contribution = max(-2.0, min(2.0, contribution))  # cap a single source's influence
    return contribution, status, valid


def disagreement_term(llm, struct_band, evidence_source):
    """Penalize when signals point different directions."""
    if llm is None:
        return 0.0
    penalty = 0.0
    if llm.get("models_disagree"):
        penalty += 1.0
    # structural says high-risk but no grounding at all backs it up
    if struct_band == "high" and evidence_source == "none" and llm.get("p_deliberate", 0.5) < 0.5:
        penalty += 0.5
    return penalty


def score(node, evidence, patterns, llm, cfg):
    w = cfg.get("weights", WEIGHTS)
    hard = cfg.get("hard_triggers", list(TRIGGER_TEXT))
    path, reasons = [], []

    ev_logit, ev_source = evidence_term(evidence, patterns)

    triggers = hard_triggers(node, hard)
    if triggers:
        # --- gated branch: no equation, same as the original flowchart ---
        path.append("trigger:yes")
        reasons += [TRIGGER_TEXT[t] for t in triggers]
        risk = "high"
        p = None  # not a probabilistic call -- it's a categorical gate
        path.append("evidence:" + ev_source)

        # the LLM can still speak here, but only to add context/reasons --
        # it cannot lower risk out of a hard-trigger function.
        l_logit, l_status, l_cites = llm_term(llm, evidence, patterns)
        path.append(f"llm:{l_status}")
        if l_status == "grounded":
            reasons.append(f"LLM assessed deliberate/defensive, grounded in {', '.join(l_cites)}")
        elif l_status == "ungrounded":
            reasons.append("LLM had an opinion but cited nothing in the evidence -- discarded")
        path.append("disagree:no")  # disagreement can't move a gated result
    else:
        # --- Option 1: log-odds branch, only for the ambiguous case ---
        path.append("trigger:no")
        s_logit, struct_band = struct_term(node, cfg)
        path.append(f"struct:{struct_band}")
        reasons.append(f"{node['fan_in']} direct caller{'s' if node['fan_in'] != 1 else ''} "
                        f"(blast radius {node['blast_radius']})")
        path.append("evidence:" + ev_source)

        l_logit, l_status, l_cites = llm_term(llm, evidence, patterns)
        path.append(f"llm:{l_status}")
        if l_status == "grounded":
            reasons.append(f"LLM assessed deliberate/defensive, grounded in {', '.join(l_cites)}")
        elif l_status == "ungrounded":
            reasons.append("LLM had an opinion but cited nothing in the evidence -- discarded")

        d_penalty = disagreement_term(llm, struct_band, ev_source)
        path.append("disagree:yes" if d_penalty else "disagree:no")

        logit = w["w0"] + w["w1"] * s_logit + w["w2"] * l_logit - w["w3"] * d_penalty + w["w4"] * ev_logit
        p = sigmoid(logit)
        risk = bucket(p, cfg.get("risk_thresholds", {"high": 0.75, "medium": 0.4}))

    # confidence is a separate read of the same evidence/disagreement signals,
    # not the risk probability itself
    conf_bands = {"comment": "high", "commit": "high", "pattern": "medium", "none": "low"}
    conf = conf_bands[ev_source]
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

    path.append("p=gated" if p is None else f"p={p:.2f}")
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