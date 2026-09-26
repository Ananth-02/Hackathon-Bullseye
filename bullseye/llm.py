"""Optional LLM layer. Works with any OpenAI-compatible chat endpoint (GLM, DeepSeek, ...).

The LLM never sets the risk score. It writes the explanation, and may say that
code "looks deliberate", which the scorer uses to raise risk by one level.
A second model, if configured, is asked the same question; disagreement lowers
confidence. Without configuration the whole tool runs offline.

Environment:
  BULLSEYE_LLM_URL    e.g. https://api.deepseek.com/v1/chat/completions
  BULLSEYE_LLM_KEY, BULLSEYE_LLM_MODEL
  BULLSEYE_LLM2_URL / BULLSEYE_LLM2_KEY / BULLSEYE_LLM2_MODEL   (optional second opinion)
  BULLSEYE_LLM_EXTRA / BULLSEYE_LLM2_EXTRA  optional JSON merged into the request body,
      e.g. '{"thinking": {"type": "disabled"}}' to switch off slow reasoning modes

Any of these may instead be written as KEY=VALUE lines in a `.env` file at the
project root. Real environment variables take precedence.
"""
import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are helping a new engineer understand legacy embedded C before changing it.
Function `{name}` from `{file}` (lines {start}-{end}).
Static-analysis facts: {facts}
Callers: {callers}
Comments found as evidence: {evidence}

```c
{code}
```

Reply with JSON only:
{{"purpose": "one sentence",
  "why": "2-3 sentences on why it is written this way; cite line numbers; say 'unknown' if the code gives no reason",
  "looks_deliberate": "yes|no|unsure  (yes = contains code that looks redundant or odd but is probably protecting against hardware, timing or concurrency problems)",
  "open_questions": ["max 3 questions an engineer should answer before changing it"]}}"""


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path=ENV_FILE):
    """KEY=VALUE lines from .env; a real environment variable always wins."""
    try:
        text = path.read_text()
    except OSError:                        # no .env, or unreadable: nothing to do
        return
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):     # so the README's export lines can be pasted in
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) > 1 and v[0] in "'\"" and v[-1] == v[0]:
            v = v[1:-1]                    # strip one layer of matching quotes
        os.environ.setdefault(k.strip(), v)


def _cfg(prefix):
    url, key, model = (os.environ.get(f"{prefix}_{k}") for k in ("URL", "KEY", "MODEL"))
    if not (url and model):
        return None
    url = url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"          # accept a base URL as well as the full endpoint
    extra = json.loads(os.environ.get(f"{prefix}_EXTRA") or "{}")
    return (url, key, model, extra)


def configured():
    _load_dotenv()
    return [c for c in (_cfg("BULLSEYE_LLM"), _cfg("BULLSEYE_LLM2")) if c]


def _extract_json(txt):
    """First balanced {...} object in txt, or None.

    Counts braces instead of matching a greedy regex: models often wrap the JSON
    in commentary that contains braces of its own, and `{.*}` then runs to the
    last one and fails to parse. Braces inside strings do not count.
    """
    start = txt.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(txt)):
            c = txt[i]
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
                        return json.loads(txt[start:i + 1])
                    except ValueError:
                        break            # malformed: fall through to the next '{'
        start = txt.find("{", start + 1)
    return None


# statuses worth another try: rate limiting and transient gateway/server faults.
# The gonka network answers 429 "out of capacity" routinely, so without this a
# whole run degrades to errors even though the key and model are fine.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_after(err, cap=60):
    """Seconds the server asked us to wait, if it said so and it is sane."""
    try:
        return min(float(err.headers.get("Retry-After", "")), cap)
    except (AttributeError, TypeError, ValueError):
        return None


def _call(cfg, prompt, timeout=120, attempts=5):
    url, key, model, extra = cfg
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 900,
                       "messages": [{"role": "user", "content": prompt}], **extra}).encode()
    headers = {"Content-Type": "application/json",
               **({"Authorization": f"Bearer {key}"} if key else {})}
    delay = 4.0
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                msg = json.load(r)["choices"][0]["message"]
            txt = msg.get("content") or msg.get("reasoning_content") or ""
            return _extract_json(txt)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == attempts:
                raise
            wait = _retry_after(e) or delay
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == attempts:
                raise
            wait = delay
        # jitter, or every worker in the pool comes back at the same instant
        time.sleep(wait + random.uniform(0, 1.0))
        delay = min(delay * 2, 60)


class LLM:
    def __init__(self, cache_path=".bullseye_llm_cache.json"):
        self.models = configured()
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    @property
    def mode(self):
        return f"{len(self.models)} model(s)" if self.models else "offline"

    def review_many(self, items, workers=6, progress=print):
        """items: list of (node, evidence). Runs requests in parallel, returns {id: review}."""
        out, done = {}, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.review, n, e): n["fn"].id for n, e in items}
            for fut in concurrent.futures.as_completed(futs):
                out[futs[fut]] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(items):
                    progress(f"LLM reviewed {done}/{len(items)}")
        return out

    def review(self, node, evidence):
        if not self.models:
            return None
        f = node["fn"]
        facts = {k: v for k, v in node["notes"].items() if v}
        prompt = PROMPT.format(name=f.name, file=f.file, start=f.line_start, end=f.line_end,
                               facts=json.dumps(facts)[:1500], callers=", ".join(node["callers"][:15]) or "none found",
                               evidence=json.dumps([e["text"] for e in evidence])[:1500], code=f.code[:7000])
        results = []
        for cfg in self.models:
            k = hashlib.sha1((cfg[2] + prompt).encode()).hexdigest()
            if k not in self.cache:
                try:
                    self.cache[k] = _call(cfg, prompt)
                except Exception as e:  # network / quota / bad JSON: degrade, never crash the run
                    self.cache[k] = {"error": str(e)[:200]}
            results.append(self.cache[k])
        good = [r for r in results if r and "error" not in r]
        if not good:
            return None
        verdicts = {str(r.get("looks_deliberate", "unsure")).lower() for r in good}
        return {**good[0], "models_disagree": len(good) > 1 and len(verdicts) > 1}

    def save(self):
        if self.models:
            # failures (quota, rate limit, network) are cached for this run only, so that
            # fixing the cause and re-running retries them instead of reusing the error
            keep = {k: v for k, v in self.cache.items() if not (v and "error" in v)}
            self.cache_path.write_text(json.dumps(keep))


def check():
    """Send one tiny request to each configured model and report what came back."""
    models = configured()
    if not models:
        print("No model configured. Set BULLSEYE_LLM_URL, BULLSEYE_LLM_KEY and BULLSEYE_LLM_MODEL")
        print(f"as environment variables, or as KEY=VALUE lines in {ENV_FILE}")
        return 1
    code = 0
    for i, cfg in enumerate(models, 1):
        try:
            r = _call(cfg, 'Reply with JSON only: {"ok": true}', timeout=120)
            print(f"model {i} ({cfg[2]} at {cfg[0]}): OK, replied {r}")
        except Exception as e:
            code = 1
            detail = e.read().decode()[:300] if hasattr(e, "read") else ""
            print(f"model {i} ({cfg[2]} at {cfg[0]}): FAILED: {e} {detail}")
    return code
