"""Step 1: parse C sources with tree-sitter and pull out raw facts.

Nothing here judges risk. It only records what is in the code: functions, the
calls they make, comments, macros, inline assembly, loops and assignments.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_c
from tree_sitter import Language, Parser

C = Language(tree_sitter_c.language())
IDENT = re.compile(r"[A-Za-z_]\w*")
ASM = re.compile(r"(?:__asm(?:__)?|\basm)\s*(?:volatile|__volatile__)?\s*\((.*?)\)\s*;", re.S)
HEX_ADDR = re.compile(r"0x([0-9A-Fa-f]{6,8})\b")
C_KEYWORDS = {"if", "else", "while", "for", "do", "switch", "case", "return", "sizeof", "goto", "defined"}


@dataclass
class Macro:
    name: str
    values: list = field(default_factory=list)
    idents: set = field(default_factory=set)
    called: set = field(default_factory=set)      # identifiers followed by "(" in the macro text
    hw_register: bool = False                      # macro IS a hardware register (volatile + address)
    writes: set = field(default_factory=set)       # identifiers the macro assigns to
    asm: list = field(default_factory=list)
    file: str = ""
    line: int = 0


@dataclass
class Function:
    name: str
    file: str
    line_start: int
    line_end: int
    static: bool
    code: str
    calls: list = field(default_factory=list)      # direct calls by name (functions or macros)
    idents: set = field(default_factory=set)       # every identifier used in the body
    indirect_calls: int = 0                         # calls through pointers: cannot be resolved statically
    assign_lhs: list = field(default_factory=list)  # (line, text) of every assignment target
    loops: list = field(default_factory=list)       # (line, empty_body, cond, body, kind, has_update)
    asm: list = field(default_factory=list)         # (line, asm text)
    doc_comment: tuple = None                       # (line, text) comment right above the function
    inner_comments: list = field(default_factory=list)
    isr_registrations: list = field(default_factory=list)  # (registering call, identifier passed)


def _text(node):
    return node.text.decode("utf8", "replace")


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _fn_name(decl):
    while decl is not None:
        if decl.type == "function_declarator":
            inner = decl.child_by_field_name("declarator")
            if inner is not None and inner.type == "identifier":
                return _text(inner)
            decl = inner
        elif decl.type in ("pointer_declarator", "parenthesized_declarator", "attributed_declarator"):
            decl = decl.child_by_field_name("declarator") or (decl.named_children[0] if decl.named_children else None)
        else:
            return None
    return None


def _is_empty_body(body):
    if body is None:
        return False
    if body.type == "expression_statement" and _text(body).strip() == ";":
        return True
    if body.type == "compound_statement":
        return all(c.type == "comment" for c in body.named_children)
    return False


DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^)]*\))?", re.M)


def parse_macros(text, rel):
    """Read #defines straight from the text. tree-sitter stops at a comment inside a
    multi-line macro, which is exactly where FreeRTOS hides its register writes."""
    out = []
    for m in DEFINE.finditer(text):
        i, lines = m.end(), []
        while True:
            j = text.find("\n", i)
            j = len(text) if j < 0 else j
            line = text[i:j]
            lines.append(line.rstrip("\\ \t\r"))
            if not line.rstrip().endswith("\\") or j >= len(text):
                break
            i = j + 1
        v = "\n".join(lines)
        v = re.sub(r"/\*.*?\*/|//[^\n]*", " ", v, flags=re.S)
        mac = Macro(m.group(1), file=rel, line=text[:m.start()].count("\n") + 1)
        mac.values.append(v)
        mac.idents |= set(IDENT.findall(v))
        mac.called |= set(re.findall(r"([A-Za-z_]\w*)\s*\(", v))
        mac.hw_register = bool(HEX_ADDR.search(v)) and ("volatile" in v or bool(re.search(r"\*\s*\)", v)))
        mac.writes |= set(re.findall(r"([A-Za-z_]\w*)\s*(?:\|=|&=|\^=|(?<![=!<>])=(?!=))", v))
        mac.asm += ASM.findall(v)
        out.append(mac)
    return out


def parse_file(path: Path, rel: str, isr_register_calls: set, fn_macros=()):
    src = path.read_bytes()
    tree = Parser(C).parse(src)
    root = tree.root_node
    functions, decls = [], []
    macros = parse_macros(src.decode("utf8", "replace"), rel)
    comments = [(c.start_point[0] + 1, c.end_point[0] + 1, _text(c)) for c in _walk(root)
                if c.type == "comment" and re.search(r"[A-Za-z]{3}", _text(c))]
    comments.sort()

    for node in _walk(root):
        t = node.type
        if t == "declaration":
            for d in node.named_children:
                if d.type in ("function_declarator", "pointer_declarator"):
                    n = _fn_name(d)
                    if n:
                        decls.append((n, node.start_point[0] + 1))

        elif t == "function_definition":
            decl = node.child_by_field_name("declarator")
            name = _fn_name(decl)
            if name in fn_macros:  # the real name is the macro's first argument
                m = re.search(r"\(\s*([A-Za-z_]\w*)", _text(decl)[len(name):])
                name = m.group(1) if m else name
            body = node.child_by_field_name("body")
            if not name or body is None or name in C_KEYWORDS:  # macro soup can make `if (...) {` look like a definition
                continue
            static = any(c.type == "storage_class_specifier" and _text(c) == "static" for c in node.children)
            s, e = node.start_point[0] + 1, node.end_point[0] + 1
            f = Function(name, rel, s, e, static, _text(node))
            for n in _walk(body):
                if n.type == "identifier":
                    f.idents.add(_text(n))
                elif n.type == "call_expression":
                    fn = n.child_by_field_name("function")
                    if fn is not None and fn.type == "identifier":
                        callee = _text(fn)
                        f.calls.append((callee, n.start_point[0] + 1))
                        if callee in isr_register_calls:
                            args = n.child_by_field_name("arguments")
                            for a in (args.named_children if args else []):
                                for idn in IDENT.findall(_text(a)):
                                    f.isr_registrations.append((callee, idn))
                    elif fn is not None and (fn.type in ("field_expression", "pointer_expression", "subscript_expression")
                                             or (fn.type == "parenthesized_expression" and re.search(r"^\(\s*\*|->|\[", _text(fn)))):
                        f.indirect_calls += 1
                elif n.type == "assignment_expression":
                    lhs = n.child_by_field_name("left")
                    if lhs is not None:
                        f.assign_lhs.append((n.start_point[0] + 1, _text(lhs)))
                elif n.type in ("while_statement", "do_statement", "for_statement"):
                    b = n.child_by_field_name("body")
                    cond = n.child_by_field_name("condition")
                    upd = n.child_by_field_name("update") if n.type == "for_statement" else None
                    f.loops.append((n.start_point[0] + 1, _is_empty_body(b),
                                    _text(cond) if cond is not None else "", _text(b) if b is not None else "",
                                    n.type, upd is not None))
                elif n.type == "gnu_asm_expression":
                    f.asm.append((n.start_point[0] + 1, _text(n)))
            if not f.asm:  # fall back to regex when the asm was parsed as something else
                for m in ASM.finditer(f.code):
                    f.asm.append((s + f.code[:m.start()].count("\n"), m.group(1)))
            # comment directly above the function (allow a short gap for attributes / prototypes)
            above = [c for c in comments if c[1] < s and c[1] >= s - 3]
            if above:
                f.doc_comment = (above[-1][0], above[-1][2])
            f.inner_comments = [(c[0], c[2]) for c in comments if s <= c[0] <= e]
            functions.append(f)

    header_docs = {}
    if rel.endswith(".h"):
        stripped = src.decode("utf8", "replace")
        text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), stripped, flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"^[ \t]*#.*?(?<!\\)$", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.M | re.S)
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\((?:[^;{}()]|\([^;{}()]*\))*\)\s*(?:[A-Z_][A-Z0-9_]*\s*)*;", text):
            decls.append((m.group(1), text[:m.start()].count("\n") + 1))
        for name, line in decls:
            above = [c for c in comments if line - 4 <= c[1] < line]
            if above:
                header_docs[name] = (above[-1][0], above[-1][2])
    return functions, macros, [d[0] for d in decls], header_docs


def collect_files(root: Path, include_globs, exclude_globs):
    files = set()
    for g in include_globs:
        files |= {p for p in root.glob(g) if p.is_file() and p.suffix in (".c", ".h")}
    for g in exclude_globs:
        files -= set(root.glob(g))
    return sorted(files)
