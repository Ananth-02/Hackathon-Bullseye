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
  └─ llm.py       optional: any OpenAI-compatible model (GLM, DeepSeek). Explains, never scores.
  └─ score.py     the scoring flowchart, recording the route each function took
        ↓
  data JSON  (the parser ↔ viewer contract)  →  viewer.py  →  bullseye.html
```

### The scoring flowchart

1. **Hard trigger?** Runs in interrupt context, writes a hardware register, critical section / interrupt masking, busy-wait or hardware delay, only ever runs inside a lock, services the watchdog → risk HIGH.
2. Otherwise **direct callers**: 15 or more → HIGH, 5–14 → MEDIUM, under 5 → LOW.
3. **Public API?** → at least MEDIUM.
4. **LLM says it looks deliberate / defensive?** → raise risk one level (never lowers it).
5. **Evidence for the why**: reason-giving comment or commit → confidence HIGH; known defensive pattern → MEDIUM; nothing → LOW.
6. **Blind spots?** Calls through function pointers inside, the function is passed as a pointer somewhere, or two LLMs disagree → lower confidence one level.

Thresholds and triggers live in the platform config, not in code.

### Platform configs ("input about the digital platform")

`bullseye/platforms/*.json` describe what is actually built and how the hardware looks: which files and port, hardware address ranges, register naming, which symbols mean interrupt masking or delays, which calls register interrupt handlers (e.g. `intConnect`, `excHookAdd`), which files are public headers. Adding a new target is writing one of these.

- `freertos-cm4f`: FreeRTOS kernel, GCC ARM_CM4F port, heap_4.
- `cfs-mcp750-vxworks`: NASA cFS OSAL + PSP for the MCP750 PowerPC board on VxWorks. Root is the folder holding the `osal` and `PSP` repos.

### Connecting an LLM (GLM and/or DeepSeek)

Any OpenAI-compatible chat endpoint works. A base URL or the full `/chat/completions` URL are both accepted.

```bash
# model 1: DeepSeek
export BULLSEYE_LLM_URL=https://api.deepseek.com
export BULLSEYE_LLM_KEY=<your DeepSeek key>
export BULLSEYE_LLM_MODEL=deepseek-flash

# model 2 (optional second opinion; disagreement lowers confidence): GLM on Z.ai
export BULLSEYE_LLM2_URL=https://api.z.ai/api/paas/v4          # mainland China: https://open.bigmodel.cn/api/paas/v4
export BULLSEYE_LLM2_KEY=<your Z.ai key>
export BULLSEYE_LLM2_MODEL=glm-5.1

# optional: extra request fields, e.g. switch off slow reasoning modes
export BULLSEYE_LLM2_EXTRA='{"thinking": {"type": "disabled"}}'

python -m bullseye llm-check                  # one tiny request per model, prints OK or the error
python -m bullseye analyze ../repos/FreeRTOS-Kernel --platform freertos-cm4f --label "FreeRTOS kernel" --git --llm-top 60 -o data/freertos.json
```

Windows PowerShell: `$env:BULLSEYE_LLM_URL="https://api.deepseek.com"` and so on.

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
