"""Adapter: drive llm_new.py's hybrid scorer from Bullseye's own call graph.

llm_new.py was written against networkx, but its `Explainer.explain()` takes
plain dicts and an optional metrics dict, so no graph library is needed here.
This module translates a Bullseye node into what llm_new expects, and
translates the result back into the shape `score.py` consumes.

Why both scorers run:

  llm_new gives a *continuous* change_confidence (0-100) built from a static
  signal scan, call-graph facts and the model's own judgment, with every
  deduction recorded in score_breakdown.

  score.py gives the *categorical* risk / confidence pair the viewer's
  flowchart is built around, and it is the thing that decides the review
  queue.

They are not redundant: score.py needs a grounded P(deliberate) from a model,
which is exactly what llm_new now produces and validates. So llm_new feeds
score.py rather than replacing it, and its hybrid number rides along for
display.

The class below deliberately mirrors the interface of `llm.LLM` (models, mode,
review_many, save), so `pipeline.py` only has to change which one it imports.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:               # llm_new.py lives at the repo root
    sys.path.insert(0, str(_ROOT))


def _signal_ids(labels):
    """Static-scan labels -> the short ids a citation can name."""
    try:
        import llm_new
        return sorted({llm_new.SIGNAL_IDS[l] for l in (labels or [])
                       if l in llm_new.SIGNAL_IDS})
    except Exception:
        return []


def _node_metrics(node, total_nodes):
    """Bullseye node -> the metrics dict llm_new.graph_score() expects.

    Bullseye stores blast_radius as a count of transitive callers; llm_new
    wants it as a share of the graph, so it is divided here rather than
    silently handing a count to something that multiplies it by 20.
    """
    flags = node["flags"]
    return {
        "fan_in": node["fan_in"],
        "fan_out": len(node["callees"]),
        "transitive_callers": node["blast_radius"],
        "blast_radius": round(node["blast_radius"] / max(1, total_nodes - 1), 2),
        "is_critical_root": bool(flags.get("isr_context") or flags.get("watchdog")),
        "critical_path_via": sorted(node["callers"])[:5] if flags.get("isr_context") else [],
        "calls_hardware": bool(flags.get("writes_hw_register") or node["hw_lines"]),
        "in_cycle": node["fn"].id in node["callees"],
    }


class HybridLLM:
    """Same surface as llm.LLM, backed by llm_new's static+graph+LLM scorer."""

    def __init__(self, platform_info="", total_nodes=1):
        self.total_nodes = total_nodes
        self.insights = {}          # id -> FunctionInsight as a dict
        self._explainer = None
        self._error = None
        try:
            import llm_new
            self._llm_new = llm_new
            self._explainer = llm_new.Explainer(platform_info=platform_info)
            self.models = [(None, None, self._explainer.backend.id, {})]
        except Exception as e:      # no key, no backend, bad config: stay offline
            self._llm_new = None
            self.models = []
            self._error = f"{type(e).__name__}: {e}"

    @property
    def mode(self):
        if self.models:
            return f"hybrid via {self.models[0][2]}"
        return f"offline ({self._error})" if self._error else "offline"

    def review_many(self, items, workers=None, progress=print):
        """items: [(node, evidence)]. Returns {function id: review dict}."""
        if not self.models:
            return {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = workers or int(os.getenv("BULLSEYE_WORKERS", "6"))
        out, done = {}, 0

        def job(node, evidence):
            f = node["fn"]
            callers = {c: "" for c in node["callers"][:6]}
            callees = {c: "" for c in node["callees"][:8]}
            ins = self._explainer.explain(
                name=f.name, code=f.code, file=f.file,
                callers=callers, callees=callees,
                metrics=_node_metrics(node, self.total_nodes),
                evidence_items=evidence,
                node_patterns=node["patterns"],
            )
            return f.id, ins

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, n, e) for n, e in items]
            for fut in as_completed(futs):
                fid, ins = fut.result()
                self.insights[fid] = self._llm_new.asdict(ins)
                out[fid] = self._to_review(ins)
                done += 1
                if done % 10 == 0 or done == len(items):
                    progress(f"LLM reviewed {done}/{len(items)}")
        return out

    @staticmethod
    def _to_review(ins):
        """FunctionInsight -> the dict score.py's llm_term() reads.

        p_deliberate is already pinned to 0.5 by llm_new when no citation
        resolved, and the cites list has had unresolvable ids stripped, so
        score.py's own grounding check agrees with llm_new's by construction.
        """
        if ins.error and ins.llm_confidence is None:
            return None
        return {
            "purpose": ins.purpose,
            "why": ins.why_it_exists,
            "p_deliberate": ins.p_deliberate if ins.p_deliberate is not None else 0.5,
            "cites": list(ins.cites),
            "models_disagree": False,        # single model; kept for score.py
            # The signal ids the static scan really found. score.py re-checks
            # citations against this itself rather than trusting llm_new's
            # verdict, so the two graders stay independent.
            "signals": _signal_ids(ins.evidence),
            "open_questions": list(ins.hazards)[:3],
            # hybrid extras, for display only
            "change_confidence": ins.change_confidence,
            "risk_level": ins.risk_level,
            "criticality": ins.criticality,
            "static_score": ins.static_score,
            "graph_score": ins.graph_score,
            "llm_confidence": ins.llm_confidence,
            "score_breakdown": list(ins.score_breakdown),
            "grounded": ins.grounded,
            "cites_rejected": list(ins.cites_rejected),
        }

    def save(self):
        """llm_new caches each answer as its own file as it goes."""
        return None
