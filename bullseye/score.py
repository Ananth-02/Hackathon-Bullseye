"""Step 4: the scoring flowchart from the team doc, as code.

Every decision is appended to `path`, so the viewer can light up the exact
route a function took through the flowchart.

v2: the LLM's contribution is now a calibrated probability with a required
evidence pointer, combined into a log-odds score alongside the structural
signal instead of jumping risk a whole tier on a bare yes/no. See
`llm_term()` for the grounding rule that prevents an ungrounded claim from
moving the score.

Implementation notes for the revised scoring flow:

1. Split the LLM's influence out of a binary jump into a calibrated,
   evidence-gated equation (Option 1: log-odds combination). Originally, the
   LLM answered a yes/no ("does this look deliberate?") that bumped risk a
   full tier. That's guessable — an LLM can say "yes" with no basis and move
   the result just as much as one with real evidence. Now the LLM outputs a
   probability (`p_deliberate`) that only enters the score if it can cite
   something real in evidence or patterns; if it can't, the probability is
   clamped to 0.5 (no information) before it's combined with the structural
   signal. This is the actual "prevent it from guessing" mechanism you asked
   for.

2. Split hard triggers back out as a categorical gate, separate from the
   equation. First pass wrongly folded hard triggers (ISR context, register
   writes, etc.) into the same continuous log-odds sum as fan-in — diluting a
   qualitative "this is dangerous to guess about" fact into a quantity that
   could get averaged down. Restored: `hard_triggers()` short-circuits
   straight to `risk="high"` before the equation ever runs, matching the
   original flowchart. The LLM can still add context/reasons on a gated
   function, but can't lower its risk.

3. Restored `floor_triggers` (dropped by mistake in the first rewrite).
   `external_interface` was sitting in the default hard-trigger list, so every
   public function gated straight to high regardless of anything else —
   confirmed on real test data (`public_get_temperature` came out high/low-
   confidence for no real reason). Fixed by giving `external_interface` its own
   default list (`DEFAULT_FLOOR_TRIGGERS`) that raises a floor on `p`
   (guarantees at least medium) instead of forcing a full gate.
   `DEFAULT_HARD_TRIGGERS` now only contains the genuinely hardware/timing-
   critical flags.

4. Citation IDs switched from list-position to stable identifiers.
   `llm_term()` used to validate a citation like `"comment:0"` against the
   index in whatever list happened to be passed in. If evidence got re-sorted
   or re-fetched between when the LLM saw it and when `score()` ran, a citation
   could silently resolve to the wrong item — a false positive worse than an
   honest "ungrounded" result. Now citations key on the comment's real source
   line (`"comment:42"`), a commit's short SHA (`"commit:a1b2c3d4e5"`), and —
   new — a pattern's stable short ID (`"pattern:barrier_asm"`) instead of its
   full description string.

5. Minor: `open_questions()` updated to read `node["patterns"][0]["text"]`
   instead of a plain string, matching the new pattern shape from `graph.py`.
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

# Hard triggers gate straight to "high" (see hard_triggers()). Floor triggers
# instead guarantee a MINIMUM of "medium" without forcing "high" -- this is
# where external_interface belongs: a public function deserves attention by
# default, but "it's public" alone isn't the same category of danger as
# "it masks interrupts" or "it writes a hardware register". Restored here
# after the v2 rewrite accidentally dropped this distinction, which made
# every public function gate to "high" via the old default hard_triggers list.
DEFAULT_HARD_TRIGGERS = ["isr_context", "writes_hw_register", "critical_section",
                          "busy_wait", "locked_context", "watchdog"]
DEFAULT_FLOOR_TRIGGERS = ["external_interface"]

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

    Citation IDs are keyed on stable identifiers, not list position: a
    comment by its source line ("comment:42"), a commit by its short sha
    ("commit:a1b2c3d4e5"), a pattern by the short id graph.py now attaches
    to it ("pattern:barrier_asm"). This survives evidence being re-sorted,
    filtered, or re-fetched between when the LLM saw it and when score()
    runs -- an index-based id would silently point at the wrong item.
    """
    if llm is None:
        return 0.0, "offline", []

    real_evidence_ids = {f"comment:{e['line']}" for e in evidence if e["type"] == "comment"}
    real_evidence_ids |= {f"commit:{e['commit']}" for e in evidence if e["type"] == "commit"}
    real_pattern_ids = {f"pattern:{p['id']}" for p in patterns}
    # A function that nothing documents still has verifiable facts found in its
    # own code. Without these, the LLM budget -- deliberately aimed at the
    # least-understood functions -- could never cite anything, and every review
    # of exactly the code this tool exists to explain would be discarded.
    real_signal_ids = {f"signal:{s}" for s in llm.get("signals", [])}
    valid = [c for c in llm.get("cites", [])
             if c in real_evidence_ids or c in real_pattern_ids or c in real_signal_ids]

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
    hard = cfg.get("hard_triggers", DEFAULT_HARD_TRIGGERS)
    path, reasons = [], []

    ev_logit, ev_source = evidence_term(evidence, patterns)

    triggers = hard_triggers(node, hard)
    if triggers:
        # --- gated branch: no equation, same as the original flowchart ---
        path.append("trigger:yes")
        reasons += [TRIGGER_TEXT[t] for t in triggers]
        risk = "high"
        p = None  # not a probabilistic call -- it's a categorical gate

        # Fan-in and public-API status are facts about the function whether or
        # not they decided the risk, so record them here too: a reader of the
        # flowchart wants to see them. They are tagged informational (the path
        # carries p=gated) so the viewer can show they did not drive the score.
        _, struct_band = struct_term(node, cfg)
        path.append(f"struct:{struct_band}")
        reasons.append(f"{node['fan_in']} direct caller{'s' if node['fan_in'] != 1 else ''} "
                        f"(blast radius {node['blast_radius']})")
        gated_floors = [t for t in cfg.get("floor_triggers", DEFAULT_FLOOR_TRIGGERS)
                        if node["flags"].get(t)]
        path.append("floor:yes" if gated_floors else "floor:no")
        if gated_floors:
            reasons += [TRIGGER_TEXT[t] + ": floor at MEDIUM (already HIGH by trigger)"
                        for t in gated_floors]

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

        thresholds = cfg.get("risk_thresholds", {"high": 0.75, "medium": 0.4})
        floors = [t for t in cfg.get("floor_triggers", DEFAULT_FLOOR_TRIGGERS) if node["flags"].get(t)]
        if floors:
            path.append("floor:yes")
            reasons += [TRIGGER_TEXT[t] + ": floor at MEDIUM" for t in floors]
            floor_p = thresholds.get("medium", 0.4) + 0.01
            if p < floor_p:
                p = floor_p
        else:
            path.append("floor:no")

        risk = bucket(p, thresholds)

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
        q.append(f"The code contains a known defensive pattern ({node['patterns'][0]['text']}). Which failure was it added for?")
    return q[:3]
