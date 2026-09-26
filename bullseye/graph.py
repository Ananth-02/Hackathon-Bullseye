"""Step 2: turn raw facts into a dependency graph with per-function flags.

Macros are expanded so that e.g. taskENTER_CRITICAL() becomes a real edge to
vPortEnterCritical(), and a register write hidden inside portYIELD() is still
seen as a hardware write by every function that uses portYIELD().
"""
import fnmatch
import re
from collections import defaultdict, deque

HEX_ADDR = re.compile(r"0x([0-9A-Fa-f]{6,8})\b")


class MacroTable:
    def __init__(self, macros):
        self.by_name = defaultdict(list)
        for m in macros:
            self.by_name[m.name].append(m)

    def __contains__(self, name):
        return name in self.by_name

    def expand(self, names, depth=6):
        """Return {identifier: macro chain that led to it} for everything reachable through macros."""
        reached = {}
        queue = deque((n, ()) for n in names if n in self.by_name)
        while queue:
            name, chain = queue.popleft()
            if len(chain) >= depth:
                continue
            chain2 = chain + (name,)
            for m in self.by_name[name]:
                for idn in m.idents:
                    if idn not in reached and idn not in names:
                        reached[idn] = chain2
                        if idn in self.by_name:
                            queue.append((idn, chain2))
        return reached

    def any(self, name, attr):
        return any(getattr(m, attr) for m in self.by_name.get(name, []))


def build(functions, macros, public_decls, header_docs, cfg):
    mt = MacroTable(macros)
    hw_macros = {m.name for m in macros if m.hw_register}
    ranges = [(int(a, 16), int(b, 16)) for a, b in cfg.get("hw_address_ranges", [])]
    reg_field = re.compile(cfg["register_field_regex"]) if cfg.get("register_field_regex") else None
    mmio = re.compile(cfg["mmio_write_regex"]) if cfg.get("mmio_write_regex") else None
    barrier = re.compile(cfg.get("barrier_asm_regex", r"$^"), re.I)
    irqmask = re.compile(cfg.get("irq_mask_asm_regex", r"$^"), re.I)
    isr_re = re.compile(cfg.get("isr_name_regex", r"$^"))
    critical = set(cfg.get("critical_markers", []))
    delays = set(cfg.get("delay_functions", []))
    wd_ident = re.compile(cfg.get("watchdog_ident_regex", r"$^"), re.I)
    lock_enter = set(cfg.get("lock_enter", []))
    lock_exit = set(cfg.get("lock_exit", []))

    def in_hw_range(text):
        for h in HEX_ADDR.findall(text):
            v = int(h, 16)
            if any(a <= v <= b for a, b in ranges):
                return True
        return False

    def is_hw_target(text):
        if any(re.search(r"\b%s\b" % re.escape(h), text) for h in hw_macros if h in text):
            return True
        if "volatile" in text and (in_hw_range(text) or HEX_ADDR.search(text) or re.search(r"\*\s*\(\s*\(?\s*volatile", text)):
            return True
        return bool((reg_field and reg_field.search(text)) or (mmio and mmio.search(text)))

    # ---- node ids: plain name when unique, file::name when a static name repeats
    by_name = defaultdict(list)
    for f in functions:
        by_name[f.name].append(f)
    for f in functions:
        f.id = f.name if len(by_name[f.name]) == 1 else f"{f.file}::{f.name}"

    def resolve(name, from_file):
        cands = by_name.get(name, [])
        if len(cands) <= 1:
            return cands
        same = [c for c in cands if c.file == from_file]
        if same:
            return same[:1]
        public = [c for c in cands if not c.static]
        return public[:1] if len(public) == 1 else cands

    nodes = {}
    edges = []               # (caller_id, callee_id, kind, via_macro)
    edge_lines = defaultdict(set)   # (caller_id, callee_id) -> call-site lines
    regions = {}             # caller_id -> list of (start_line, end_line) inside a lock
    isr_roots = {}           # id -> reason
    public = set(public_decls)

    for f in functions:
        called_direct = {c for c, _ in f.calls}
        expanded = mt.expand(f.idents | called_direct)
        flags, notes, hw_lines = {}, defaultdict(list), set()

        # calls: direct, and calls hidden inside macros
        seen = set()
        call_lines = defaultdict(list)
        for c, line in f.calls:
            call_lines[c].append(line)
        for c, line in f.calls:
            for tgt in resolve(c, f.file):
                if tgt is not f:
                    edge_lines[(f.id, tgt.id)].add(line)
                if tgt is not f and tgt.id not in seen:
                    seen.add(tgt.id)
                    edges.append((f.id, tgt.id, "call", None))
        for idn, chain in expanded.items():
            if idn in by_name and any(idn in m.called for m in mt.by_name.get(chain[-1], [])):
                for tgt in resolve(idn, f.file):
                    if tgt is not f:
                        edge_lines[(f.id, tgt.id)].update(call_lines.get(chain[0], []))
                    if tgt is not f and tgt.id not in seen:
                        seen.add(tgt.id)
                        edges.append((f.id, tgt.id, "call", chain[0]))
        # lock regions: lines between an enter marker and its matching exit marker
        events = sorted([(l, 1, c) for c, l in f.calls if c in lock_enter] + [(l, -1, c) for c, l in f.calls if c in lock_exit])
        depth, start, reg = 0, None, []
        for l, d, c in events:
            if d > 0 and depth == 0:
                start = (l, c)
            depth = max(0, depth + d)
            if depth == 0 and start:
                reg.append((start[0], l, start[1]))
                start = None
        regions[f.id] = reg
        # function names used as values (callbacks, task entry points, ISR registration)
        for idn in f.idents - called_direct:
            if idn in by_name and idn != f.name:
                for tgt in resolve(idn, f.file):
                    edges.append((f.id, tgt.id, "ref", None))

        # hardware register writes (direct, or inside a macro the function uses)
        for line, lhs in f.assign_lhs:
            if is_hw_target(lhs) or any(mt.any(i, "hw_register") for i in re.findall(r"[A-Za-z_]\w*", lhs)):
                hw_lines.add(line)
                notes["writes_hw_register"].append(f"line {line}: {lhs.strip()[:60]}")
        for idn, chain in expanded.items():
            for m in mt.by_name.get(chain[-1], []):
                if idn in m.writes and (idn in hw_macros or is_hw_target(idn)):
                    notes["writes_hw_register"].append(f"{idn} via {chain[0]}")
                    break
        flags["writes_hw_register"] = bool(notes["writes_hw_register"])

        # inline assembly: interrupt masking and memory barriers
        own_asm = [(l, a) for l, a in f.asm]
        asm_texts = list(own_asm)
        for idn in f.idents | called_direct | set(expanded):
            for m in mt.by_name.get(idn, []):
                asm_texts += [(None, a) for a in m.asm]
        masking = [a for a in asm_texts if irqmask.search(a[1])]
        barriers = [a for a in asm_texts if barrier.search(a[1])]
        for l, _ in masking + barriers:
            if l:
                hw_lines.add(l)

        # critical sections
        crit = sorted((called_direct | f.idents | set(expanded)) & critical)
        if crit:
            notes["critical_section"].append("uses " + ", ".join(crit[:3]))
        if masking:
            notes["critical_section"].append("masks interrupts in inline assembly")
        flags["critical_section"] = bool(crit or masking)

        # busy-wait loops and timing delays
        for line, empty, cond, body, kind, has_update in f.loops:
            polls_hw = is_hw_target(cond) or "volatile" in cond or any(h in cond for h in hw_macros)
            nop_only = "nop" in body and body.count(";") <= 3
            spin = empty and not has_update and not re.search(r"\+\+|--|[^=!<>]=[^=]", cond)
            if spin or nop_only or (polls_hw and body.count(";") <= 2):
                hw_lines.add(line)
                notes["busy_wait"].append(f"line {line}: loop {'waits on ' + cond.strip()[:50] if cond else 'with empty body'}")
        d = sorted(called_direct & delays)
        if d:
            notes["busy_wait"].append("calls " + ", ".join(d))
        flags["busy_wait"] = bool(notes["busy_wait"])

        wd = sorted({i for i in (f.idents | called_direct | {f.name}) if wd_ident.search(i)})
        flags["watchdog"] = bool(wd)
        if wd:
            notes["watchdog"].append("uses " + ", ".join(wd[:3]))

        flags["external_interface"] = (not f.static) and f.name in public
        if flags["external_interface"]:
            notes["external_interface"].append("declared in a public header")

        if isr_re.search(f.name):
            isr_roots[f.id] = "name marks it as an interrupt handler / ISR-safe API"
        for reg, idn in f.isr_registrations:
            for tgt in by_name.get(idn, []):
                isr_roots[tgt.id] = f"registered as an interrupt / exception handler via {reg}() in {f.name}"

        patterns = []
        if any(barrier.search(a) for _, a in own_asm):
            patterns.append("memory / instruction barrier (dsb, isb, sync…)")
        lhs_seq = [l for _, l in f.assign_lhs]
        if any(a == b and is_hw_target(a) for a, b in zip(lhs_seq, lhs_seq[1:])):
            patterns.append("same register written twice in a row")
        if any(h in hw_macros or is_hw_target(h) for h in re.findall(r"\(\s*void\s*\)\s*([A-Za-z_]\w*)\s*;", f.code)):
            patterns.append("dummy read of a register")
        if any(re.search(r"\bnop\b", a, re.I) for _, a in own_asm):
            patterns.append("nop timing padding")

        nodes[f.id] = dict(fn=f, flags=flags, notes=dict(notes), hw_lines=sorted(hw_lines),
                           patterns=patterns, macro_hits=sorted(set(c[0] for c in expanded.values()))[:12])

    # ---- adjacency
    callers, callees, refs_in = defaultdict(set), defaultdict(set), defaultdict(set)
    via = {}
    for a, b, kind, v in edges:
        if kind == "call":
            callees[a].add(b)
            callers[b].add(a)
            if v:
                via[(a, b)] = v
        else:
            refs_in[b].add(a)

    # ---- interrupt context: roots plus everything they (transitively) call
    isr_ctx = dict(isr_roots)
    queue = deque(isr_roots)
    while queue:
        cur = queue.popleft()
        for nxt in callees[cur]:
            if nxt not in isr_ctx:
                isr_ctx[nxt] = f"called from interrupt context via {cur.split('::')[-1]}"
                queue.append(nxt)
    for nid, n in nodes.items():
        n["flags"]["isr_context"] = nid in isr_ctx
        if nid in isr_ctx:
            n["notes"]["isr_context"] = [isr_ctx[nid]]

    # ---- locked context (fixed point)
    def in_region(caller, lines):
        return bool(lines) and all(any(a < l < b for a, b, _ in regions.get(caller, [])) for l in lines)

    locked = {}
    changed = True
    while changed:
        changed = False
        for nid in nodes:
            if nid in locked or not callers[nid] or refs_in[nid]:
                continue
            why = []
            for c in callers[nid]:
                lines = edge_lines.get((c, nid), set())
                if c in locked:
                    why.append(f"{c.split('::')[-1]} (itself only runs locked)")
                elif in_region(c, lines):
                    lk = next(r[2] for r in regions[c] if any(r[0] < l < r[1] for l in lines))
                    why.append(f"{c.split('::')[-1]} inside {lk}")
                else:
                    why = None
                    break
            if why:
                locked[nid] = why
                changed = True
    for nid, n in nodes.items():
        n["flags"]["locked_context"] = nid in locked
        if nid in locked:
            n["notes"]["locked_context"] = ["every caller holds a lock: " + "; ".join(locked[nid][:3])]

    # ---- blast radius: every function that can reach this one
    def ancestors(nid):
        seen, q = set(), deque([nid])
        while q:
            for p in callers[q.popleft()]:
                if p not in seen:
                    seen.add(p)
                    q.append(p)
        seen.discard(nid)
        return seen

    for nid, n in nodes.items():
        n["callers"] = sorted(callers[nid])
        n["callees"] = sorted(callees[nid])
        n["refs_in"] = sorted(refs_in[nid])
        n["fan_in"] = len(callers[nid])
        n["blast_radius"] = len(ancestors(nid))
        n["via"] = {b: v for (a, b), v in via.items() if a == nid}
        n["header_doc"] = header_docs.get(n["fn"].name)
    return nodes
