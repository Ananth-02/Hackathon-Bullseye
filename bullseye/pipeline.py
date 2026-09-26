"""Glue: repo + platform config -> data JSON (the parser <-> viewer contract)."""
import datetime
import json
import subprocess
from pathlib import Path

from . import evidence as ev
from .extract import collect_files, parse_file
from .graph import build
from .llm import LLM
from .score import open_questions, score

PLATFORM_DIR = Path(__file__).parent / "platforms"


def load_platform(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = PLATFORM_DIR / f"{name_or_path}.json"
    return json.loads(p.read_text())


def _git_head(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _git_root(path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return Path(out) if out else None
    except Exception:
        return None


def analyze(root, platform, use_git=False, git_top=80, llm_top=60, label=None, progress=print):
    root = Path(root).resolve()
    cfg = load_platform(platform)
    files = collect_files(root, cfg["include"], cfg.get("exclude", []))
    progress(f"parsing {len(files)} files")
    functions, macros, decls, hdocs = [], [], [], {}
    isr_reg = set(cfg.get("isr_register_calls", []))
    for fp in files:
        rel = str(fp.relative_to(root))
        fs, ms, ds, hd = parse_file(fp, rel, isr_reg, set(cfg.get("function_macros", [])))
        # #if variants of the same function in one file: keep the first definition
        known = {(f.file, f.name) for f in functions}
        functions += [f for f in fs if (f.file, f.name) not in known and not known.add((f.file, f.name))]
        macros += ms
        if any(fp.match(g) or Path(rel).match(g) for g in cfg.get("public_headers", [])):
            decls += ds
            hdocs.update(hd)
    progress(f"{len(functions)} functions, {len(macros)} macros")
    nodes = build(functions, macros, decls, hdocs, cfg)

    llm = LLM()
    # rank for optional expensive steps: triggers first, then fan-in
    order = sorted(nodes.values(), key=lambda n: (-sum(n["flags"].values()), -n["fan_in"]))
    git_ids = {n["fn"].id for n in order[:git_top]} if use_git else set()

    # pass 1: evidence and a rule-only score for every function
    prelim = {}
    for n in nodes.values():
        f = n["fn"]
        evidence = ev.comment_evidence(f)
        if f.id in git_ids:
            g = _git_root((root / f.file).parent)
            if g:
                rel_to_git = str((root / f.file).resolve().relative_to(g))
                evidence += ev.git_evidence(g, rel_to_git, f.line_start, f.line_end)
        prelim[f.id] = (evidence, score(n, evidence, n["patterns"], None, cfg))

    # pass 2: LLM only for the functions an engineer should look at first (cost and time)
    reviews = {}
    if llm.models and llm_top:
        R, C = {"high": 0, "medium": 1, "low": 2}, {"low": 0, "medium": 1, "high": 2}
        pre = lambda n: prelim[n["fn"].id][1]
        # half the budget explains the riskiest, least-understood code;
        # the other half checks lower-rated code where an LLM "looks deliberate" can still raise the risk
        explain = sorted((n for n in nodes.values() if pre(n)[0] == "high"),
                         key=lambda n: (C[pre(n)[1]], -n["blast_radius"]))[:(llm_top + 1) // 2]
        hunt = sorted((n for n in nodes.values() if pre(n)[0] != "high"),
                      key=lambda n: (C[pre(n)[1]], -len(n["patterns"]) - bool(n["fn"].asm), -n["blast_radius"]))[:llm_top - len(explain)]
        rank = explain + hunt
        progress(f"asking the LLM about {len(explain)} high-risk and {len(hunt)} lower-rated functions")
        reviews = llm.review_many([(n, prelim[n["fn"].id][0]) for n in rank], progress=progress)

    out = []
    for n in nodes.values():
        f = n["fn"]
        evidence = prelim[f.id][0]
        review = reviews.get(f.id)
        risk, conf, path, reasons, down, triggers = score(n, evidence, n["patterns"], review, cfg)
        out.append({
            "id": f.id, "name": f.name, "file": f.file, "lines": [f.line_start, f.line_end],
            "static": f.static,
            "purpose": (review or {}).get("purpose") or ev.purpose(f, n["header_doc"]),
            "why": (review or {}).get("why"),
            "risk": risk, "confidence": conf, "score_path": path,
            "reasons": reasons, "confidence_notes": down, "triggers": triggers,
            "flags": n["flags"], "flag_notes": n["notes"], "patterns": n["patterns"],
            "evidence": [{k: v for k, v in e.items() if k != "score"} for e in evidence],
            "open_questions": (review or {}).get("open_questions") or open_questions(n, triggers, conf, evidence),
            "callers": n["callers"], "callees": n["callees"], "refs_in": n["refs_in"], "via": n["via"],
            "fan_in": n["fan_in"], "blast_radius": n["blast_radius"],
            "indirect_calls": f.indirect_calls, "hw_lines": n["hw_lines"],
            "llm": None if review is None else {"looks_deliberate": review.get("looks_deliberate"),
                                                "models_disagree": review.get("models_disagree")},
            "code": f.code,
        })
    llm.save()
    out.sort(key=lambda x: x["id"])
    counts = {}
    for x in out:
        counts[f"{x['risk']}/{x['confidence']}"] = counts.get(f"{x['risk']}/{x['confidence']}", 0) + 1
    return {
        "meta": {"label": label or root.name, "platform": cfg["name"], "platform_note": cfg.get("description"),
                 "root": root.name, "commit": _git_head(root), "files": len(files),
                 "generated": datetime.datetime.now().isoformat(timespec="minutes"),
                 "llm": llm.mode + (f", top {llm_top} functions" if llm.models else ""), "llm_models": [m[2] for m in llm.models], "git_evidence": use_git, "thresholds": cfg.get("thresholds"),
                 "counts": counts},
        "functions": out,
    }
