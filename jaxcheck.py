"""Which JAX set works on this GPU? Staged checks, each in its own process, hang-safe.

    python jaxcheck.py --label my-wheel-set                     # all stages
    python jaxcheck.py --label my-wheel-set --stages matmul scatter_set
    XLA_FLAGS=... python jaxcheck.py --label with-flags          # try an XLA flag
    python jaxcheck.py --sweep --stages mjx_reset mjx_step       # the flag configurations in SWEEP

Why each stage runs in a subprocess: on ROCm a GPU memory fault aborts the HIP queue but can leave
the process hanging instead of exiting. So every stage streams its output to a log, and is recorded as
  ok        the stage printed its result and exited 0
  fail      it exited non-zero (last error line kept)
  fault     a GPU fault signature appeared: the process is killed at once
  timeout   no result within --timeout seconds: killed
Results: results/results.jsonl (one row per stage) and results/README.md (the table).
The mjx_* stages need `mujoco_playground` (pip install playground) and use its Op3Joystick env.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("results")
N_ENVS = 16
FAULT_SIGNS = ("HSA_STATUS_ERROR", "Memory Fault Error", "ROCM_ERROR_ILLEGAL_ADDRESS")
SWEEP = {  # flags jaxlib 0.9.2 accepted; kept to re-run on new stacks
    "no-determinism-expander": "--xla_gpu_enable_scatter_determinism_expander=false",
}


def stage_code(stage: str) -> str:
    """Python source for one stage; prints 'OK <seconds>' on success."""
    prelude = "import time, jax, jax.numpy as jp, numpy as np\nt = time.perf_counter()\n"
    env = ("from mujoco_playground import registry\n"
           "cfg = registry.get_default_config('Op3Joystick'); cfg.impl = 'jax'\n"
           "env = registry.load('Op3Joystick', config=cfg)\n"
           f"keys = jax.random.split(jax.random.PRNGKey(0), {N_ENVS})\n")
    body = {
        "devices": "y = jp.zeros(1); print('devices', jax.devices(), flush=True)",
        "matmul": ("a = np.random.default_rng(0).standard_normal((512, 512), dtype=np.float32)\n"
                   "y = jax.jit(lambda x: jp.sin(x) @ x.T)(jp.asarray(a))\n"
                   "assert np.allclose(np.asarray(y), np.sin(a) @ a.T, rtol=1e-3, atol=1e-2), 'wrong result'"),
        "scatter_set": "y = jax.jit(lambda a, i: a.at[i].set(1.0))(jp.zeros(1024), jp.arange(0, 1024, 7))",
        "scatter_add": "y = jax.jit(lambda a, i: a.at[i].add(1.0))(jp.zeros(1024), jp.arange(1024) % 13)",
        "scatter_2d": "y = jax.jit(lambda a, i: a.at[i, 1:4].set(2.0))(jp.zeros((64, 8)), jp.arange(0, 64, 3))",
        "scatter_vmap": "y = jax.jit(jax.vmap(lambda a, i: a.at[i].set(1.0)))(jp.zeros((16, 100)), jp.arange(16) * 5)",
        "scatter_oob": ("y = jax.jit(lambda a, i: a.at[i].set(1.0, mode='drop'))(jp.zeros(1024), jp.array([0, 5, 1024, 10**6, 2**30]))\n"
                        "assert float(y.sum()) == 2.0, y.sum()"),
        "mjx_reset": env + "y = jax.jit(jax.vmap(env.reset))(keys)",
        "mjx_step": env + ("s = jax.jit(jax.vmap(env.reset))(keys)\n"
                           "y = jax.jit(jax.vmap(env.step))(s, jp.zeros((len(keys), env.action_size)))"),
    }[stage]
    return prelude + body + "\njax.block_until_ready(y)\nprint('OK', round(time.perf_counter() - t, 1), flush=True)\n"


STAGES = ["devices", "matmul", "scatter_set", "scatter_add", "scatter_2d", "scatter_vmap", "scatter_oob",
          "mjx_reset", "mjx_step"]


def run_stage(stage: str, label: str, timeout_s: float, xla_flags: str | None = None, code: str | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{label}-{stage}.log"
    flags = os.environ.get("XLA_FLAGS", "") if xla_flags is None else xla_flags
    env = {**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "rocm"), "XLA_FLAGS": flags}
    t = time.monotonic()
    status = None
    with log.open("w") as out:
        p = subprocess.Popen([sys.executable, "-u", "-c", code or stage_code(stage)], env=env,
                             stdout=out, stderr=subprocess.STDOUT)
        while (rc := p.poll()) is None:
            if any(s in log.read_text(errors="replace") for s in FAULT_SIGNS):
                status = "fault"
            elif time.monotonic() - t > timeout_s:
                status = "timeout"
            if status:
                p.kill()
                p.wait()
                rc = None
                break
            time.sleep(0.5)
    text = log.read_text(errors="replace")
    if status is None:
        ok = rc == 0 and any(l.startswith("OK ") for l in text.splitlines())
        status = "ok" if ok else "fault" if any(s in text for s in FAULT_SIGNS) else "fail"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    errors = [l for l in lines if any(s in l.lower() for s in ("error", "fault", "check failed", "unknown flag"))] or lines[-1:]
    return {"label": label, "stage": stage, "status": status, "exit_code": rc,
            "seconds": round(time.monotonic() - t, 1), "error": errors[-1][:200] if status != "ok" and errors else "",
            "xla_flags": flags, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def render(rows: list[dict]) -> str:
    lines = ["# jaxcheck results", "", "| time | label | stage | status | seconds | XLA_FLAGS | last error |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['time'][:16]} | {r['label']} | {r['stage']} | {r['status']} | {r['seconds']} "
                     f"| `{r['xla_flags'] or '-'}` | {r['error'].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--label", help="names this configuration (XLA_FLAGS from the environment)")
    who.add_argument("--sweep", action="store_true", help="run every configuration in SWEEP")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--timeout", type=float, default=600, help="seconds per stage (MJX compiles take minutes)")
    p.add_argument("--pause", type=float, default=20, help="seconds after a failed stage; the GPU needs a moment after a fault")
    args = p.parse_args(argv)
    results = OUT / "results.jsonl"
    for label, flags in (SWEEP.items() if args.sweep else [(args.label, None)]):
        for stage in args.stages:
            row = run_stage(stage, label, args.timeout, flags)
            print(f"{label:<28} {stage:<13} {row['status']:<8} {row['seconds']:>7}s  {row['error']}", flush=True)
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if row["status"] != "ok":
                time.sleep(args.pause)
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    (OUT / "README.md").write_text(render(rows))


if __name__ == "__main__":
    main()
