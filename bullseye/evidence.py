"""Step 3: evidence for *why* the code is the way it is.

Sources, strongest first: a comment that states a reason, a commit message that
names a fix or workaround, a recognisable defensive code pattern. Every item
keeps a line number or commit id so the UI can cite it.
"""
import re
import subprocess

STRONG = re.compile(
    r"\b(because|otherwise|must|required|requires|ensure[sd]?|prevent|workaround|errata|erratum|"
    r"in case|so that|cannot|can't|do not|don't|necessary|important|avoid|guarantee|reason|"
    r"to stop|to protect|protect|race|deadlock|corrupt|critical|assumes?|note)\b", re.I)
COMMIT_NOISE = re.compile(r"\b(misra|warnings?|typos?|spelling|format\w*|style|lint|coverity|docs?|comments?|headers?|license|copyright|cspell|build|cmake|examples?|tests?|coverage|cleanup|clean up|rename\w*|refactor\w*|parenthes\w*|unnecessary|typedefs?)\b", re.I)
COMMIT_WORDS = re.compile(r"\b(workaround|errata|erratum|race|hang|hangs|crash\w*|deadlock|corrupt\w*|overflow|null|timing|interrupt|isr|stack|priority|reset|watchdog|bug|incorrect|wrong|regression|revert|lock\w*|determinism|alignment|exception)\b", re.I)


def clean_comment(text):
    text = re.sub(r"^\s*/\*+|\*+/\s*$", "", text.strip())
    text = re.sub(r"^\s*//+", "", text)
    lines = [re.sub(r"^\s*\*\s?", "", l).rstrip() for l in text.splitlines()]
    return lines


def _sentences(lines, first_line):
    """Yield (line_no, sentence) for a cleaned comment."""
    buf, start = [], first_line
    for i, l in enumerate(lines):
        if not l.strip():
            if buf:
                yield start, " ".join(buf)
                buf = []
            continue
        if not buf:
            start = first_line + i
        buf.append(l.strip())
    if buf:
        yield start, " ".join(buf)


def comment_evidence(fn, limit=4):
    items = []
    blocks = ([fn.doc_comment] if fn.doc_comment else []) + fn.inner_comments
    for line, text in blocks:
        for sline, para in _sentences(clean_comment(text), line):
            for sent in re.split(r"(?<=[.!?])\s+", para):
                hits = STRONG.findall(sent)
                if hits and len(sent) > 25 and not sent.lower().startswith(("copyright", "spdx", "permission is hereby")):
                    items.append({"type": "comment", "line": sline, "text": sent[:260],
                                  "score": len(hits) + (3 if re.search(r"workaround|errata|erratum|otherwise|because", sent, re.I) else 0)})
    items.sort(key=lambda x: -x["score"])
    seen, out = set(), []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"])
            out.append(it)
    return out[:limit]


def purpose(fn, header_doc):
    """First real sentence of the API doc (header) or of the comment above the definition."""
    for src in (header_doc, fn.doc_comment):
        if not src:
            continue
        lines, keep, skip = clean_comment(src[1]), [], False
        for l in lines:
            s = l.strip()
            if re.match(r"(@code|<pre>)", s):
                skip = True
            if s.startswith(("@param", "@return", "\\defgroup", "\\ingroup", "Example usage")):
                break
            s = re.sub(r"-{4,}|\*{4,}|^[@\\]brief\s*", " ", s).strip()
            if not skip and s and not re.match(r"^[\w_]+\.\s*h\b|^\w+\.c$", s):
                keep.append(s)
            if re.match(r"(@endcode|</pre>)", s):
                skip = False
        text = re.sub(r"<[^>]+>", "", " ".join(keep)).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text.split()) < 4:   # e.g. "configSUPPORT_DYNAMIC_ALLOCATION" left after an #endif
            continue
        m = re.match(r"(.{20,300}?[.!?])(\s|$)", text)
        if m:
            return m.group(1)
        if len(text) > 20:
            return text[:240]
    return None


def git_evidence(repo_root, file, start, end, timeout=20, cache={}):
    """Commit subjects that touched these lines and mention a fix/workaround."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "blame", "-L", f"{start},{end}", "--line-porcelain", "--", file],
                             capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    commits, cur = {}, None
    for line in out.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            cur = line[:40]
        elif line.startswith("summary ") and cur:
            commits[cur] = re.sub(r"^(Fix|Fixes|Partial|Resolves?)\s+#\d+[,:]?\s*", "", line[8:])
    items = []
    for sha, subj in commits.items():
        if COMMIT_WORDS.search(subj) and not COMMIT_NOISE.search(subj):
            items.append({"type": "commit", "commit": sha[:10], "text": subj[:200]})
    return items[:3]
