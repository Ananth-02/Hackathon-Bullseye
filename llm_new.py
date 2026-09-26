"""
bullseye_llm.py — LLM + hybrid risk scoring layer for Bullseye.

For every function node in the call graph, Bullseye computes a
change_confidence (0 = do not touch, 100 = safe to change) from three parts:

  1. STATIC score  — deterministic: red-flag patterns found in the code
  2. GRAPH score   — deterministic: how many things depend on it, and whether
                     it sits on a critical path (ISR, control loop, watchdog…)
  3. LLM score     — the model's judgment after reading code + context

  final = w_static*static + w_graph*graph + w_llm*llm   (then safety caps)

Every point deducted is recorded in `score_breakdown`, so any score can be
explained line by line.

Backends (swappable, same interface):
  - ClaudeBackend        : Anthropic API (demo)
  - OllamaBackend        : local open-weight model (on-prem story for SAAB)
  - OpenAICompatBackend  : any OpenAI-style endpoint (e.g. DeepSeek)

Setup:
  pip install anthropic networkx requests
  Put settings in .env next to this file (see .env), or export them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ------------------------------------------------------------------- .env

def load_env(path: str = ".env") -> None:
    """Tiny .env loader (no dependency). Real env vars win over the file."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


# ---------------------------------------------------------------- result type

@dataclass
class FunctionInsight:
    name: str
    # --- LLM explanation
    purpose: str = ""
    why_it_exists: str = ""
    usage_context: str = ""
    criticality: str = "unknown"      # safety | mission | business | low | unknown
    hazards: list[str] = field(default_factory=list)
    reasoning: str = ""
    # --- scores (0-100, higher = safer to change)
    static_score: int = 100
    graph_score: int | None = None    # None when analysed without a graph
    llm_confidence: int | None = None # None when the LLM call failed
    p_deliberate: float | None = None  # model's probability the oddity is deliberate
    cites: list[str] = field(default_factory=list)      # citations that resolved
    cites_rejected: list[str] = field(default_factory=list)  # citations that did not
    grounded: bool = False            # at least one citation resolved
    change_confidence: int = 50       # FINAL hybrid score
    risk_level: str = "MEDIUM"        # LOW | MEDIUM | HIGH (risk of changing)
    # --- explainability
    evidence: list[str] = field(default_factory=list)          # signals found
    graph_metrics: dict = field(default_factory=dict)
    score_breakdown: list[str] = field(default_factory=list)   # every deduction
    error: str | None = None


# ============================================================ 1. STATIC SCORE
# label: (regex, penalty). Start at 100, subtract each signal found once.

SIGNALS: dict[str, tuple[str, int]] = {
    "interrupt handler / ISR":
        (r"\b(ISR|IRQHandler|__interrupt|interrupt)\b", 25),
    "workaround / errata comment":
        (r"(?i)(workaround|errata|hack|do not remove|don't remove|quirk|bug in|fixme|xxx)", 25),
    "raw memory-mapped address":
        (r"\*\s*\(\s*(volatile\s+)?\w+\s*\*\s*\)\s*0x[0-9A-Fa-f]+", 20),
    "interrupt enable/disable (critical section)":
        (r"\b(__disable_irq|__enable_irq|cli\(\)|sei\(\)|DISABLE_INTERRUPTS|ENABLE_INTERRUPTS)\b", 20),
    "watchdog interaction":
        (r"(?i)(watchdog|\bwdt|iwdg|kick_?dog)", 20),
    "volatile access (hardware register / shared state)":
        (r"\bvolatile\b|\b[A-Z][A-Z0-9]*->[A-Z]{2,}\b", 15),   # also catches ADC1->DR style
    "busy-wait / explicit delay (timing assumption)":
        (r"\b(delay|udelay|mdelay|HAL_Delay|usleep|nop|__NOP)\s*\(|while\s*\(.+?\)\s*(\{\s*\}|;)", 15),
    "retry loop / redundancy":
        (r"(?i)\b(retry|retries|redundan|voting|double[-_ ]?check)\b", 15),
    "preprocessor platform switch":
        (r"#\s*if(def)?\s+\w*(HW|REV|BOARD|PLATFORM|CPU|TARGET)\w*", 10),
    "magic hex constant":
        (r"\b0x[0-9A-Fa-f]{4,}\b", 5),
}

# Short, stable ids for the signals above. A citation names one of these
# rather than the full label, which contains slashes and parentheses that a
# model cannot be relied on to reproduce character for character.
#
# These matter more than they look: the LLM budget is deliberately spent on
# the code that NOTHING documents, so comments, commits and defensive-pattern
# citations are all unavailable there by definition. Without a citable fact
# derived from the code itself, a grounding rule would discard every review of
# exactly the functions the tool exists to explain.
SIGNAL_IDS = {
    "interrupt handler / ISR": "isr",
    "workaround / errata comment": "errata_comment",
    "raw memory-mapped address": "mmio_address",
    "interrupt enable/disable (critical section)": "irq_mask",
    "watchdog interaction": "watchdog",
    "volatile access (hardware register / shared state)": "volatile_access",
    "busy-wait / explicit delay (timing assumption)": "busy_wait",
    "retry loop / redundancy": "retry_loop",
    "preprocessor platform switch": "platform_switch",
    "magic hex constant": "magic_hex",
}

HW_SIGNALS = {"raw memory-mapped address",
              "volatile access (hardware register / shared state)"}
CRITICAL_SIGNALS = {"interrupt handler / ISR", "watchdog interaction"}


def scan_signals(code: str) -> list[str]:
    return [label for label, (pat, _) in SIGNALS.items() if re.search(pat, code)]


def static_score(evidence: list[str]) -> tuple[int, list[str]]:
    score, why = 100, []
    for label in evidence:
        pen = SIGNALS[label][1]
        score -= pen
        why.append(f"static  -{pen:<3} {label}")
    return max(0, score), why


# ============================================================= 2. GRAPH SCORE

CRITICAL_NAME = re.compile(r"(?i)(isr|irq|handler|^main$|control|safety|watchdog|fault|reset)")


def find_critical_roots(G, evidence_by_node: dict) -> set:
    """Nodes that are critical by themselves: ISR/watchdog code or critical names."""
    return {n for n in G.nodes
            if CRITICAL_SIGNALS & set(evidence_by_node.get(n, []))
            or CRITICAL_NAME.search(str(n))}


def graph_metrics(G, n, critical_roots: set, evidence_by_node: dict) -> dict:
    import networkx as nx
    ancestors = nx.ancestors(G, n)        # everything that (transitively) calls n
    descendants = nx.descendants(G, n)    # everything n (transitively) calls
    others = max(1, G.number_of_nodes() - 1)
    crit_via = sorted(str(a) for a in ancestors & critical_roots)
    return {
        "fan_in": G.in_degree(n),
        "fan_out": G.out_degree(n),
        "transitive_callers": len(ancestors),
        "blast_radius": round(len(ancestors) / others, 2),   # share of graph affected
        "is_critical_root": n in critical_roots,
        "critical_path_via": crit_via[:5],
        "calls_hardware": any(HW_SIGNALS & set(evidence_by_node.get(d, []))
                              for d in descendants),
        "in_cycle": n in ancestors,     # recursion / mutual recursion
    }


def graph_score(m: dict) -> tuple[int, list[str]]:
    score, why = 100, []

    def dock(pen, text):
        nonlocal score
        if pen > 0:
            score -= pen
            why.append(f"graph   -{pen:<3} {text}")

    dock(min(20, 4 * m["fan_in"]), f"{m['fan_in']} direct caller(s)")
    dock(round(20 * m["blast_radius"]),
         f"{m['transitive_callers']} functions depend on it ({int(m['blast_radius']*100)}% of graph)")
    if m["is_critical_root"]:
        dock(20, "is itself a critical entry point (ISR / watchdog / control / main)")
    elif m["critical_path_via"]:
        dock(20, f"on critical path via {', '.join(m['critical_path_via'])}")
    if m["calls_hardware"]:
        dock(10, "calls code that touches hardware registers")
    if m["in_cycle"]:
        dock(10, "part of a call cycle (recursion)")
    return max(0, score), why


def graph_facts_text(m: dict) -> str:
    if not m:
        return ""
    lines = [f"- direct callers: {m['fan_in']}, transitive callers: {m['transitive_callers']}"
             f" ({int(m['blast_radius']*100)}% of the codebase)",
             f"- calls {m['fan_out']} function(s); reaches hardware-touching code: {m['calls_hardware']}"]
    if m["is_critical_root"]:
        lines.append("- this function is itself a critical entry point")
    elif m["critical_path_via"]:
        lines.append(f"- on a critical path via: {', '.join(m['critical_path_via'])}")
    if m["in_cycle"]:
        lines.append("- part of a recursive call cycle")
    return "\n".join(lines)


# ============================================================ 3. COMBINE

def _weights() -> tuple[float, float, float]:
    raw = os.getenv("BULLSEYE_WEIGHTS", "0.4,0.3,0.3")
    try:
        ws, wg, wl = (float(x) for x in raw.split(","))
    except ValueError:
        ws, wg, wl = 0.4, 0.3, 0.3
    return ws, wg, wl


SAFETY_CAP = int(os.getenv("BULLSEYE_SAFETY_CAP", "35"))


def combine(static: int, graph: int | None, llm: int | None,
            evidence: list[str], criticality: str,
            criticality_grounded: bool = False) -> tuple[int, list[str]]:
    ws, wg, wl = _weights()
    parts = [(static, ws, "static")]
    if graph is not None:
        parts.append((graph, wg, "graph"))
    if llm is not None:
        parts.append((llm, wl, "llm"))
    total_w = sum(w for _, w, _ in parts)            # re-normalise if a part is missing
    final = round(sum(s * w for s, w, _ in parts) / total_w)
    formula = " + ".join(f"{w/total_w:.2f}×{name}({s})" for s, w, name in parts)
    why = [f"combine  {formula} = {final}"]

    # Safety handling. A hard clamp to SAFETY_CAP made every safety function
    # land on the same number, so the review queue could not order its most
    # dangerous items -- two functions scoring 44 and 68 both came out 35.
    # Compress into [0, SAFETY_CAP] instead: safety code is still forced to the
    # top of the queue, but keeps its relative ordering.
    #
    # A detected signal (watchdog, ISR) is deterministic and always counts. The
    # LLM's own "criticality": "safety" only counts when it cited real evidence
    # -- otherwise a model could cap any score by asserting the word "safety",
    # which is the guessing this design exists to prevent.
    crit = CRITICAL_SIGNALS & set(evidence)
    llm_says_safety = (criticality == "safety") and criticality_grounded
    if crit or llm_says_safety:
        if final > SAFETY_CAP:
            reason = (", ".join(sorted(crit)) if crit
                      else "LLM rated safety-critical, grounded in cited evidence")
            compressed = round(final * SAFETY_CAP / 100.0)
            why.append(f"safety   {final} -> {compressed} "
                       f"(compressed into 0-{SAFETY_CAP}: {reason})")
            final = compressed
    elif criticality == "safety":
        why.append("safety   not applied: LLM said safety-critical but cited nothing")
    return final, why


def risk_level(conf: int) -> str:
    return "LOW" if conf >= 70 else "MEDIUM" if conf >= 40 else "HIGH"


# ================================================================== prompt

SYSTEM_PROMPT = """You are a senior embedded-systems engineer helping a new team
understand a legacy codebase BEFORE they change it. Your job is to reconstruct
intent, dependencies and risk — not to rewrite code.

Be careful: code that looks redundant may protect against a hardware anomaly
found years ago. Timing, register access, interrupt handling, watchdogs and
external interfaces deserve extra caution. If you are guessing, say so.

Reply with ONLY a JSON object, no markdown, with exactly these keys:
{
  "purpose": string,
  "why_it_exists": string,
  "usage_context": string,
  "criticality": "safety" | "mission" | "business" | "low" | "unknown",
  "hazards": [string],
  "change_confidence": integer 0-100 (0 = do not touch, 100 = safe to change),
  "p_deliberate": number 0.0-1.0 (probability the odd-looking code is deliberate;
                  0.5 means no idea),
  "cites": [string],   // ids copied verbatim from "Evidence you may cite".
                       // Cite ONLY ids listed there. An empty list is the right
                       // answer when nothing listed supports your view; an
                       // invented id is discarded and counts as no evidence.
  "reasoning": string
}"""


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n/* ...truncated... */"


def citable(evidence_items=None, patterns=None, signals=None):
    """Render the listing of ids the model may cite, plus the valid-id set.

    Ids are stable identifiers rather than list positions -- a comment by its
    source line, a commit by its short sha, a pattern by its short id -- so a
    citation survives the evidence being re-sorted or re-fetched between the
    model seeing it and the score being computed.
    """
    lines, valid = [], set()
    for e in (evidence_items or []):
        if e.get("type") == "comment" and e.get("line") is not None:
            cid = "comment:%s" % e["line"]
        elif e.get("type") == "commit" and e.get("commit"):
            cid = "commit:%s" % e["commit"]
        else:
            continue
        valid.add(cid)
        lines.append("- %s  %s" % (cid, str(e.get("text", ""))[:160]))
    for pat in (patterns or []):
        if isinstance(pat, dict) and pat.get("id"):
            cid = "pattern:%s" % pat["id"]
            valid.add(cid)
            lines.append("- %s  %s" % (cid, pat.get("text", "")))
    for label in (signals or []):
        sid = SIGNAL_IDS.get(label)
        if sid:
            cid = "signal:%s" % sid
            valid.add(cid)
            lines.append("- %s  %s (found by the static scan of this function)"
                         % (cid, label))
    return ("\n".join(lines) if lines else "- (nothing citable for this function)"), valid


def build_prompt(name: str, code: str, file: str = "",
                 callers: dict[str, str] | None = None,
                 callees: dict[str, str] | None = None,
                 evidence: list[str] | None = None,
                 platform_info: str = "",
                 graph_facts: str = "",
                 evidence_items: list | None = None,
                 node_patterns: list | None = None,
                 signals: list | None = None) -> str:
    callers = callers or {}
    callees = callees or {}
    parts = [f"## Function under analysis: `{name}`" + (f" (file: {file})" if file else "")]
    if platform_info:
        parts.append(f"## Target platform\n{platform_info}")
    parts.append(f"```c\n{_trim(code, 6000)}\n```")
    if evidence:
        parts.append("## Signals detected by static scan\n" + "\n".join(f"- {e}" for e in evidence))
    if graph_facts:
        parts.append("## Call-graph facts\n" + graph_facts)
    if callers:
        parts.append("## Called by")
        for cname, csrc in list(callers.items())[:6]:
            parts.append(f"### {cname}\n```c\n{_trim(csrc, 1200)}\n```")
    if callees:
        parts.append("## Calls")
        for cname, csrc in list(callees.items())[:8]:
            parts.append(f"### {cname}\n```c\n{_trim(csrc, 800)}\n```")
    cite_text, _ = citable(evidence_items, node_patterns, signals)
    parts.append("## Evidence you may cite\n" + cite_text)
    parts.append("Analyse the function under analysis and return the JSON.")
    return "\n\n".join(parts)


def _parse_json(text: str) -> dict:
    """First balanced {...} object. Brace counting, not a greedy regex: a model
    that wraps its JSON in commentary containing another brace would otherwise
    make `{.*}` run to the last one and fail to parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no parseable JSON object in model reply")


# ================================================================ backends

class ClaudeBackend:
    def __init__(self, model: str | None = None, max_tokens: int = 1200):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("BULLSEYE_MODEL", "claude-sonnet-5")
        self.max_tokens = max_tokens
        self.id = f"claude:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OllamaBackend:
    """Local model, nothing leaves the machine. e.g. `ollama pull qwen2.5-coder:7b`."""
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.id = f"ollama:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        import requests
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatBackend:
    """Any OpenAI-style /chat/completions endpoint (e.g. your earlier DeepSeek setup)."""
    def __init__(self):
        self.url = os.environ["BULLSEYE_LLM_URL"].rstrip("/")
        self.key = os.environ["BULLSEYE_LLM_KEY"]
        self.model = os.environ["BULLSEYE_LLM_MODEL"]
        self.id = f"openai:{self.url}:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        import random
        import time

        import requests
        # The gonka network returns 429 "out of capacity" routinely and this
        # runs several workers at once, so a single attempt loses most calls.
        delay = 4.0
        for attempt in range(1, 6):
            try:
                return self._post(system, prompt)
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code not in (408, 409, 425, 429, 500, 502, 503, 504) or attempt == 5:
                    raise
            except requests.RequestException:
                if attempt == 5:
                    raise
            time.sleep(delay + random.uniform(0, 1.0))   # jitter: workers retry together otherwise
            delay = min(delay * 2, 60)

    def _post(self, system: str, prompt: str) -> str:
        import requests
        r = requests.post(f"{self.url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}]},
                          timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def get_backend(kind: str | None = None):
    kind = (kind or os.getenv("BULLSEYE_BACKEND", "claude")).lower()
    if kind == "ollama":
        return OllamaBackend()
    if kind == "openai":
        return OpenAICompatBackend()
    return ClaudeBackend()


# ================================================================ explainer

class Explainer:
    def __init__(self, backend=None, cache_dir: str | None = None,
                 platform_info: str = ""):
        self.backend = backend or get_backend()
        self.cache = Path(cache_dir or os.getenv("BULLSEYE_CACHE_DIR", ".bullseye_cache"))
        self.cache.mkdir(exist_ok=True)
        self.platform_info = platform_info

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.backend.id + SYSTEM_PROMPT + prompt).encode()).hexdigest()[:24]
        return self.cache / f"{key}.json"

    def _ask_llm(self, prompt: str) -> dict:
        cp = self._cache_path(prompt)
        if cp.exists():                      # cached -> instant, repeatable demo
            return json.loads(cp.read_text())
        data = _parse_json(self.backend.complete(SYSTEM_PROMPT, prompt))
        cp.write_text(json.dumps(data, indent=2))
        return data

    def explain(self, name: str, code: str, file: str = "",
                callers: dict[str, str] | None = None,
                callees: dict[str, str] | None = None,
                metrics: dict | None = None,
                evidence_items: list | None = None,
                node_patterns: list | None = None) -> FunctionInsight:
        # 1. static
        evidence = scan_signals(code)
        s_score, s_why = static_score(evidence)
        ins = FunctionInsight(name=name, evidence=evidence, static_score=s_score)

        # 2. graph (only if metrics were computed by annotate_graph)
        g_why: list[str] = []
        if metrics:
            ins.graph_metrics = metrics
            ins.graph_score, g_why = graph_score(metrics)

        # 3. llm
        prompt = build_prompt(name, code, file, callers, callees, evidence,
                              self.platform_info, graph_facts_text(metrics or {}),
                              evidence_items=evidence_items, node_patterns=node_patterns,
                              signals=evidence)
        _, valid_ids = citable(evidence_items, node_patterns, evidence)
        try:
            data = self._ask_llm(prompt)
            ins.purpose = data.get("purpose", "")
            ins.why_it_exists = data.get("why_it_exists", "")
            ins.usage_context = data.get("usage_context", "")
            ins.criticality = data.get("criticality", "unknown")
            ins.hazards = list(data.get("hazards", []))
            ins.reasoning = data.get("reasoning", "")
            ins.llm_confidence = max(0, min(100, int(data.get("change_confidence", 50))))
            # Keep only citations that resolve to something really present.
            raw_cites = [str(c) for c in (data.get("cites") or [])]
            ins.cites = [c for c in raw_cites if c in valid_ids]
            ins.cites_rejected = [c for c in raw_cites if c not in valid_ids]
            ins.grounded = bool(ins.cites)
            try:
                pd = float(data.get("p_deliberate", 0.5))
            except (TypeError, ValueError):
                pd = 0.5
            # An uncited opinion carries no information, so it is pinned to 0.5
            # and cannot move anything downstream.
            ins.p_deliberate = min(1.0, max(0.0, pd)) if ins.grounded else 0.5
        except Exception as e:  # LLM down -> still score from static + graph
            ins.error = f"{type(e).__name__}: {e}"

        # 4. combine
        ins.change_confidence, c_why = combine(s_score, ins.graph_score, ins.llm_confidence,
                                               evidence, ins.criticality,
                                               criticality_grounded=ins.grounded)
        if ins.llm_confidence is None:
            llm_line = ["llm      unavailable, excluded"]
        elif ins.grounded:
            llm_line = [f"llm      {ins.llm_confidence} (model judgment, "
                        f"p_deliberate={ins.p_deliberate:.2f}, cites {', '.join(ins.cites)})"]
        else:
            rej = (f", discarded {', '.join(ins.cites_rejected)}" if ins.cites_rejected else "")
            llm_line = [f"llm      {ins.llm_confidence} (model judgment, UNGROUNDED: "
                        f"cited nothing that resolves{rej})"]
        ins.score_breakdown = s_why + g_why + llm_line + c_why
        ins.risk_level = risk_level(ins.change_confidence)
        return ins

    # ---- networkx integration -------------------------------------------
    def annotate_graph(self, G, code_attr: str = "code", file_attr: str = "file",
                       workers: int | None = None, only=None, progress: bool = True):
        """
        G: networkx.DiGraph where an edge A -> B means "A calls B" and each
        node has G.nodes[n][code_attr] = source text.

        Adds to every analysed node:
          insight (dict), change_confidence, risk_level,
          static_score, graph_score, llm_confidence
        """
        src = lambda n: G.nodes[n].get(code_attr, "") or ""

        # Phase A: static scan of the WHOLE graph (needed to find critical roots)
        evidence_by_node = {n: scan_signals(src(n)) for n in G.nodes}
        roots = find_critical_roots(G, evidence_by_node)

        # Phase B: graph metrics per node (deterministic, fast)
        metrics = {n: graph_metrics(G, n, roots, evidence_by_node) for n in G.nodes}

        # Phase C: LLM + combine, in parallel
        nodes = [n for n in (only or G.nodes) if src(n)]

        def job(n):
            return n, self.explain(
                name=str(n), code=src(n), file=G.nodes[n].get(file_attr, ""),
                callers={str(p): src(p) for p in G.predecessors(n)},
                callees={str(s): src(s) for s in G.successors(n)},
                metrics=metrics[n],
                evidence_items=G.nodes[n].get("evidence_items"),
                node_patterns=G.nodes[n].get("patterns"),
            )

        workers = workers or int(os.getenv("BULLSEYE_WORKERS", "6"))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, n) for n in nodes]
            for i, f in enumerate(as_completed(futures), 1):
                n, ins = f.result()
                d = G.nodes[n]
                d["insight"] = asdict(ins)
                d["change_confidence"] = ins.change_confidence
                d["risk_level"] = ins.risk_level
                d["static_score"] = ins.static_score
                d["graph_score"] = ins.graph_score
                d["llm_confidence"] = ins.llm_confidence
                if progress:
                    tag = f"{ins.change_confidence:3d}% {ins.risk_level:<6}"
                    if ins.error:
                        tag += " (LLM err)"
                    print(f"[{i}/{len(nodes)}] {tag} {n}")
        return G


# ================================================================ reporting

def print_report(G) -> None:
    """Riskiest functions first, with the full score breakdown."""
    rows = sorted((d for _, d in G.nodes(data=True) if "insight" in d),
                  key=lambda d: d["change_confidence"])
    for d in rows:
        ins = d["insight"]
        print(f"\n== {ins['name']}: {ins['change_confidence']}% ({ins['risk_level']} risk)")
        for line in ins["score_breakdown"]:
            print("   " + line)


def export_json(G, path: str = "bullseye_graph.json") -> None:
    """Graph + insights as JSON for the frontend (graph view / flowchart)."""
    from networkx.readwrite import json_graph
    Path(path).write_text(json.dumps(json_graph.node_link_data(G), indent=2, default=str))


# ================================================================ quick test

if __name__ == "__main__":
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("adc_read", file="adc.c", code="""
uint16_t adc_read(void) {
    ADC1->CR |= ADC_START;
    while (!(ADC1->SR & ADC_EOC)) {}
    uint16_t v = ADC1->DR;
    /* Rev B board: first sample after wake is garbage, read twice. Do not remove. */
    ADC1->CR |= ADC_START;
    while (!(ADC1->SR & ADC_EOC)) {}
    return ADC1->DR;
}""")
    G.add_node("control_loop", file="main.c", code="""
void control_loop(void) {
    uint16_t t = adc_read();
    if (t > OVERTEMP_LIMIT) shutdown_motor();
    kick_watchdog();
}""")
    G.add_node("format_log_line", file="log.c", code="""
int format_log_line(char *buf, int n, const char *msg) {
    return snprintf(buf, n, "[LOG] %s\\n", msg);
}""")
    G.add_edge("control_loop", "adc_read")

    ex = Explainer(platform_info="STM32F4, bare metal, 1 kHz control loop")
    ex.annotate_graph(G)
    print_report(G)
