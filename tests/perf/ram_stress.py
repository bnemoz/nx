"""RAM stress harness for nx: drives the real turn loop offline against the
eval stub server, pumping a large in-memory transcript and churning the
per-path structures, while a sampler thread records RSS.

Two tasks:
  A. big-transcript: 30 read_file turns, each ~100K chars into the transcript
     (NX_TOOL_RESULT_SPILL=0 keeps tool results in RAM).
  B. path-churn: 1500 write_file calls to unique paths (churns
     _fusion._FileQueue locks and session machinery).
"""
import contextlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/home/nmishra/Documents/Projects/nx")
os.environ.update(
    NX_ENV_FILE="/dev/null",
    NX_SKIP_STARTUP_PROBES="1",
    NX_LOG_REQUESTS="0",           # don't park 400KB serialized prompts in the log queue
    NX_AUTOCOMPRESS_PERCENT="0",   # never fold the transcript
    NX_APPROVAL="yolo",
    NX_TOOL_RESULT_SPILL="0",      # keep tool results in RAM (grow the transcript)
)

import nx as _m
from nx import _state as st
from nx._batch import run_batch
from nx._eval_cmd import SCENARIOS, _stub_server

OUT = Path(__file__).parent / ".nx-ram-stress"
WS = OUT / "workspace"


def current_rss_kb() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)


samples = []
stop = threading.Event()


def sampler():
    while not stop.is_set():
        samples.append((time.time(), current_rss_kb()))
        time.sleep(0.5)


def big_file(path: Path, lines: int) -> None:
    """Varied lines (identical-line dedup would collapse them)."""
    with open(path, "w") as f:
        for i in range(lines):
            f.write(f"line {i}: {i * 7919 % 999983} alpha beta gamma delta {i}\n")


def build_workspace() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    WS.mkdir(parents=True)
    big_file(WS / "big.txt", 12000)      # ~700KB of varied text
    big_file(WS / "big2.txt", 12000)


def main() -> None:
    build_workspace()
    instr_a = "RAMSTRESS-A: read the big files repeatedly until told to stop"
    instr_b = "RAMSTRESS-B: write many unique files one after another"

    # Task A: 30 read_file turns (~3MB into the transcript), then final.
    script_a = []
    for i in range(30):
        script_a.append({"tool_call": {"name": "read_file", "args": {
            "path": "big.txt" if i % 2 == 0 else "big2.txt",
            "offset": (i % 9) * 1200 + 1, "limit": 25000}}})
    script_a.append({"final": "done reading"})

    # Task B: 1500 write_file calls to unique paths (path churn), then final.
    script_b = [{"tool_call": {"name": "write_file", "args": {
        "path": f"gen/dir_{i // 250}/file_{i}.txt",
        "content": f"payload {i} " + "x" * 500}} for i in range(1500)}]
    script_b.append({"final": "done writing"})

    SCENARIOS["ram-a"] = {"title": "big transcript", "instruction": instr_a,
                          "script": script_a}
    SCENARIOS["ram-b"] = {"title": "path churn", "instruction": instr_b,
                          "script": script_b}

    srv, base_url = _stub_server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    saved = (st.ACTIVE, st.client, st.MODEL, dict(st.SOURCES), list(st.SOURCE_ORDER))
    old_cwd = os.getcwd()
    os.environ["NX_HTTP_TIMEOUT_READ"] = "30"
    try:
        src = _m._sources.Source("ram-stub", base_url, "sk-h", "ram-stub")
        src.client = _m._sources.OpenAI(base_url=base_url, api_key="sk-h")
        src._context_window = None
        src._context_window_probe_attempted = True
        st.SOURCES["ram-stub"] = src
        st.SOURCE_ORDER.append("ram-stub")
        st.ACTIVE = src
        st.client = src.client
        st.MODEL = "ram-stub"
        st.YOLO = True

        tasks = OUT / "tasks.jsonl"
        with open(tasks, "w") as f:
            f.write(json.dumps({"id": "big-transcript", "instruction": instr_a,
                                "timeout_seconds": 0}) + "\n")
            f.write(json.dumps({"id": "path-churn", "instruction": instr_b,
                                "timeout_seconds": 0}) + "\n")

        os.chdir(WS)
        results = OUT / "results.jsonl"
        t0 = time.time()
        sampler_t = threading.Thread(target=sampler, daemon=True)
        sampler_t.start()
        with open(results, "w") as rf, contextlib.redirect_stdout(rf):
            code = run_batch(["-f", str(tasks), "--max-turns", "99999",
                              "--parallel", "1", "--continue-on-error",
                              "--yolo", "--json"])
        elapsed = time.time() - t0
        stop.set()
        sampler_t.join(timeout=2)

        for line in results.read_text().splitlines():
            r = json.loads(line)
            print(f"task {r.get('id')}: status={r.get('status')} "
                  f"turns={r.get('turns')} elapsed={r.get('elapsed_seconds')}s "
                  f"exit={r.get('exit_code')} err={str(r.get('error'))[:200]}")
        peak = max((rss for _, rss in samples), default=0)
        base = samples[0][1] if samples else 0
        print(f"\nbatch exit code: {code}, wall: {elapsed:.1f}s")
        print(f"RSS start: {base/1024:.0f} MB, peak: {peak/1024:.0f} MB; timeline:")
        for k in range(0, len(samples), max(1, len(samples) // 15)):
            ts, rss = samples[k]
            print(f"  t+{ts - t0:6.1f}s  {rss/1024:7.0f} MB")
    finally:
        stop.set()
        os.chdir(old_cwd)
        srv.shutdown()
        srv.server_close()
        st.SOURCES.clear()
        st.SOURCES.update(saved[3])
        st.SOURCE_ORDER[:] = saved[4]
        st.ACTIVE, st.client, st.MODEL = saved[0], saved[1], saved[2]


if __name__ == "__main__":
    main()
