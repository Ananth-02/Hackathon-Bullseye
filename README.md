# Bullseye

Find the code in legacy embedded C that must not be changed carelessly, and show why.

Built for the Saab track at the Gothenburg Tech Week × Chalmers Hackathon 2026:
*"How can AI help a new engineering team reconstruct the system's intent, dependencies and risks before making a change?"*

For every function Bullseye gives two separate scores:

- **Risk**: how dangerous it is to change the function.
- **Confidence**: how sure the tool is about its explanation of *why* the function is written that way.

"High risk, low confidence" is the dangerous cell: code that looks load-bearing and that nothing explains. The review queue puts those first.

## Quick start

```bash
pip install -r requirements.txt
./fetch_repos.sh          # FreeRTOS kernel + NASA OSAL and PSP into ../repos
./run_demo.sh             # analyse, build bullseye.html, run the evaluation
```

Python 3.9 or newer. On 3.9 the newest `tree-sitter` binding is 0.23.2, which is
why `requirements.txt` caps `tree-sitter-c` below 0.23.5: the later grammars ship
an ABI that binding cannot load. The scripts call `python`, so it has to be on
PATH (on Windows, an Anaconda install needs `Library\bin` on PATH too, or run
them from Anaconda Prompt, otherwise `import ssl` fails and every LLM call dies
with `unknown url type: https`).

`bullseye.html` is self-contained: open it in any browser, no server needed. Prebuilt data for both codebases is in `data/`.

## Architecture

```
repo + platform config
  └─ extract.py   tree-sitter C parse: functions, calls, comments, macros (read from raw text),
  │               inline asm, loops, assignments, handler registrations
  └─ graph.py     call graph with macro expansion, interrupt-context propagation,
  │               fan-in, blast radius, hardware / critical-section / busy-wait flags
  └─ evidence.py  the "why": reason-giving comments, git blame commit messages, known patterns
  └─ llm_new.py   optional: any OpenAI-compatible model. Static signal scan + call-graph
  │               facts + model judgment, with every deduction recorded. Explains, never scores.
  └─ hybrid.py    adapts llm_new to the call graph and hands score.py a grounded probability
  └─ score.py     the scoring flowchart, recording the route each function took
        ↓
  data JSON  (the parser ↔ viewer contract)  →  viewer.py  →  bullseye.html
```

`llm.py` is the earlier single-purpose LLM layer, kept because `llm-check` uses it
to test a connection without running an analysis.

### The scoring flowchart

1. **Hard trigger?** Runs in interrupt context, writes a hardware register, critical
   section / interrupt masking, busy-wait or hardware delay, only ever runs inside a
   lock, services the watchdog → risk HIGH. This is a gate, not a term in a sum: the
   kind of code where a wrong guess is dangerous is a category, and nothing averages
   it back down.
2. Otherwise the ambiguous case is combined in **log-odds**, and the result bucketed:

   ```
   logit(p) = w₁·structure + w₂·llm − w₃·disagreement + w₄·evidence
   ```

   where *structure* is the fan-in band, *evidence* is what documents the function,
   and *llm* is the model's P(this oddity is deliberate).
3. **Public API?** → a floor of at least MEDIUM, never a jump to HIGH.
4. **Evidence for the why**: reason-giving comment or commit → confidence HIGH;
   known defensive pattern → MEDIUM; nothing → LOW.
5. **Blind spots?** Calls through function pointers inside, the function is passed as
   a pointer somewhere, or two models disagree → lower confidence one level.

Thresholds and triggers live in the platform config, not in code. The weights are
hand-picked, not fitted to data — see the evaluation below for what that costs.

### The LLM has to earn its influence

Asked whether odd-looking code is deliberate, a model can always say yes, and a
confident yes raises risk. Since a fluent guess and a real observation read the
same, the answer is not judged — the model is made to **name its source**, and the
source is checked here:

| citation | resolves against |
|---|---|
| `comment:141` | a reason-giving comment at that source line |
| `commit:8be86d4a2b` | the commit that introduced those lines |
| `pattern:barrier_asm` | a known defensive pattern found in the code |
| `signal:busy_wait` | a fact the static scan found in this function |

Citations that do not resolve are dropped. If none survive, `p_deliberate` is pinned
to 0.5 — in log-odds exactly zero — so the function is scored as though the model had
never run. The model cannot move a score by asserting something, only by pointing at
evidence that survives an independent check. `llm_new` validates when the reply
arrives and `score.py` re-derives the valid ids and checks again, so a bug in one
grader is not inherited by the other.

Citing the static-scan signals matters more than it looks: half the LLM budget goes
to the *least* understood functions, which by definition have no comment, commit or
pattern. Without a citable fact drawn from the code itself, the rule would discard
every review of exactly the code this tool exists to explain.

### How well does it work

`python -m bullseye eval data/freertos.json eval/freertos.json` compares the scores
with hand-written expectations:

| | agrees with the human label |
|---|---|
| FreeRTOS kernel | 7 / 9 |
| NASA cFS (OSAL + PSP) | 5 / 6 |

Both failures are visible in the output rather than hidden: `vListInsert` is scored
MEDIUM where a human said HIGH, and `uxTaskGetNumberOfTasks` HIGH where a human said
LOW.

On grounding, across 120 reviewed functions the model issued 42 citations and
invented **none**. 90 of those functions had nothing citable at all, and all 90 were
correctly scored without the model's input.

### Platform configs ("input about the digital platform")

`bullseye/platforms/*.json` describe what is actually built and how the hardware looks: which files and port, hardware address ranges, register naming, which symbols mean interrupt masking or delays, which calls register interrupt handlers (e.g. `intConnect`, `excHookAdd`), which files are public headers. Adding a new target is writing one of these.

- `freertos-cm4f`: FreeRTOS kernel, GCC ARM_CM4F port, heap_4.
- `cfs-mcp750-vxworks`: NASA cFS OSAL + PSP for the MCP750 PowerPC board on VxWorks. Root is the folder holding the `osal` and `PSP` repos.

### Connecting an LLM

Any OpenAI-compatible chat endpoint works. A base URL or the full `/chat/completions`
URL are both accepted. Settings go in a `.env` file at the project root (gitignored)
or in the environment; a real environment variable always wins.

```bash
# the model that reviews functions
export BULLSEYE_BACKEND=openai
export BULLSEYE_LLM_URL=https://api.gonka-api.org/v1       # or any OpenAI-compatible base URL
export BULLSEYE_LLM_KEY=<your key>
export BULLSEYE_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731

# hybrid scoring: weights for static, graph and model; and the range safety-critical
# code is compressed into so it sorts to the top of the queue but keeps its ordering
export BULLSEYE_WEIGHTS=0.4,0.3,0.3
export BULLSEYE_SAFETY_CAP=35
export BULLSEYE_WORKERS=6

# optional second opinion; disagreement between models lowers confidence
export BULLSEYE_LLM2_URL=https://api.z.ai/api/paas/v4          # mainland China: https://open.bigmodel.cn/api/paas/v4
export BULLSEYE_LLM2_KEY=<your Z.ai key>
export BULLSEYE_LLM2_MODEL=glm-5.1

# optional: extra request fields, e.g. switch off slow reasoning modes
export BULLSEYE_LLM2_EXTRA='{"thinking": {"type": "disabled"}}'

python -m bullseye llm-check                  # one tiny request per model, prints OK or the error
python -m bullseye analyze ../repos/FreeRTOS-Kernel --platform freertos-cm4f --label "FreeRTOS kernel" --git --llm-top 60 -o data/freertos.json
```

Requests retry with backoff on HTTP 429, which shared inference brokers return
routinely under load. Each answered prompt is cached under `.bullseye_cache/`, so
re-running after a scoring change costs nothing; editing the prompt invalidates it.

Windows PowerShell: `$env:BULLSEYE_LLM_URL="https://api.gonka-api.org/v1"` and so on.

Environment variables only live in the shell that set them. To keep the keys
across terminals, copy `.env.example` to `.env` in the project root and fill it
in; `bullseye/llm.py` reads it on every run. A real environment variable of the
same name overrides the file, and `.env` is gitignored. With no `.env` and no
variables set, the tool runs fully offline and the viewer says so.

`--llm-top N` limits cost: half of N goes to the riskiest, least-understood functions (to explain them), the other half to lower-rated functions with odd code (where "looks deliberate" can still raise the risk). Answers are cached in `.bullseye_llm_cache.json`, so re-runs are free. Without these variables the tool runs fully offline and the viewer says so. Model names change: check the provider's docs if a request fails with "model not found".

## Evaluation

`eval/*.json` hold labels written by hand from reading the code, before comparing with the tool. Current results (`eval/results_*.txt`):

- FreeRTOS: 7 of 9 match. `uxTaskGetNumberOfTasks` was labelled low risk by hand, but the tool is right that it is called from ISR-safe queue functions (through the `prvIncrementQueueTxLock` macro). `vListInsert` is still rated MEDIUM; the hand label assumed it only runs with interrupts masked, but the timer task calls it without a lock, so the label's reasoning was wrong, not the rule.
- cFS: 5 of 6 match (was 4 of 6 before the watchdog trigger). `CFE_PSP_GetProcessorName` is rated MEDIUM only because it is public API.

Rules added after the first evaluation, both general rather than tuned to a case: **watchdog** (any function whose name or calls mention the watchdog) and **locked context** (every call site sits between a lock-enter and lock-exit marker, or inside a function that only runs locked; FreeRTOS: 29 functions such as `prvCopyDataFromQueue`).

## Design decisions

- **Public API sets a floor of MEDIUM, not HIGH.** As a hard trigger it marked 225 of 306 FreeRTOS functions HIGH. Move `external_interface` back into `hard_triggers` in a platform config to restore that rule.
- **Analyse the whole graph, display a neighbourhood.** Blast radius uses every layer; the viewer shows 2 layers each way and the top 5 neighbours, with "show more".
- **Macros are expanded.** Embedded C hides register writes and critical sections in macros (`portYIELD`, `taskENTER_CRITICAL`); without expansion those functions look harmless.
- **`#if` variants of one function in one file:** the first definition is kept.

## Limitations

- Static analysis only: timing assumptions are inferred from code patterns, not measured.
- Needs source code. Binary input (Ghidra) is a next step.
- Register definitions outside the analysed tree are invisible. The MCP750 board registers live in the VxWorks BSP, which is not in the repo.
- Calls through function pointers are not resolved; the tool lowers confidence instead of guessing.
- cFS: cFE is not analysed, so many PSP functions show no callers.
- Comment evidence is keyword-based; an LLM gives better explanations.
- Review notes (Confirm / Correct) are stored in the browser only.
- On the MCP750 PSP the watchdog functions are stubs (their comment still says "pc-linux"), so the watchdog trigger reflects the role of the code, not what this board does.

## Layout

```
bullseye/            package: extract, graph, evidence, llm, score, pipeline, viewer, cli
bullseye/platforms/  platform configs
.env.example         template for the LLM keys (copy to .env)
data/                analysed output for both codebases
eval/                hand labels and results
bullseye.html        prebuilt viewer with both codebases
```

## License

MIT, see [LICENSE](LICENSE). This covers Bullseye itself; the codebases it
analyses (FreeRTOS kernel, NASA OSAL and PSP) are cloned by `fetch_repos.sh`,
are not part of this repository, and keep their own licenses.
