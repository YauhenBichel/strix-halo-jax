# strix-halo-jax

Run **JAX** and **MuJoCo MJX** (robotics simulation, e.g. MuJoCo Playground) on the GPU of an
**AMD Ryzen AI MAX** ("Strix Halo", Radeon 8060S, `gfx1151`) from pip wheels — **no system ROCm
install** — and a hang-safe check that tells you which wheel set works on your machine.

## The working recipe (11 September 2026)

Linux x86_64, Python 3.12, [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install --index https://repo.amd.com/rocm/whl/gfx1151/ --index-strategy unsafe-best-match \
  "jax==0.11.1" "jaxlib==0.11.1" "jax-rocm7-plugin==0.11.1" "jax-rocm7-pjrt==0.11.1" "rocm[libraries]==7.13.0"
python -c "import jax; print(jax.devices())"        # [RocmDevice(id=0)]
```

or, locked: `uv sync --extra gfx1151 --extra mjx`. Your user must be in the `render` group
(`/dev/kfd` is `root:render 0660`): `sudo usermod -aG render,video $USER`, then log in again.

That is all: the 0.11.1 plugin (from **PyPI**) finds AMD's `rocm-sdk` library wheels by itself. No
`LD_LIBRARY_PATH`, no `HSA_OVERRIDE_GFX_VERSION`.

If the GPU is shared (with a local LLM, say), set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or JAX takes
75 % of device memory at start-up.

## Compatibility matrix

Each cell from `python jaxcheck.py --label …` ([evidence](docs/evidence-2026-09-11.md)) on a Ryzen AI MAX+ 395 (64 GiB BIOS carve-out as VRAM),
Ubuntu 24.04, kernel 7.0, no system ROCm. MJX stages: `mujoco_playground` 0.2.0 `Op3Joystick`,
`impl="jax"`, 16 environments.

| Wheel set | devices | matmul | scatters | MJX reset | MJX step |
|---|---|---|---|---|---|
| **JAX 0.11.1**: plugin + PJRT 0.11.1 from PyPI, `rocm[libraries]==7.13.0` (AMD gfx1151 index), no env vars | ok | ok | ok | **ok** | **ok** |
| JAX 0.9.2: TheRock nightly `0.9.2+rocm7.14.0a20260608` + `rocm[libraries]` of the same date, with `rocm-run` | ok | ok | ok | **fault**, then the process hangs | not reached |
| JAX 0.9.1: AMD gfx1151 release index `0.9.1+rocm7.13.0` + `rocm[libraries,devel]==7.13.0` | plugin cannot load: `librocm_sysdeps_hwloc.so.5`, `librocm_sysdeps_pciaccess.so.0` not in any 7.13.0 SDK wheel ([TheRock#8163](https://github.com/ROCm/TheRock/issues/8163)) | – | – | – | – |

The 0.9.x fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `input_scatter_fusion_5`, a large fused
scatter built from MJX kinematics) matches the fused-kernel bug fixed in XLA for JAX ≥ 0.10.0
([rocm-jax#453](https://github.com/ROCm/rocm-jax/issues/453)); no XLA flag avoided it on 0.9.2. Details:
[rocm-jax#132](https://github.com/ROCm/rocm-jax/issues/132#issuecomment-5637478637).

**Please add your row** (another kernel, a system ROCm install, a newer index): [CONTRIBUTING.md](CONTRIBUTING.md).

## MJX throughput on this chip

Raw `vmap(env.step)` of `Op3Joystick`, 100 steps after compile, JAX 0.11.1, GPU shared with a resident
52 GB LLM:

| environments | GPU (env steps/s) | one CPU device (env steps/s) |
|---|---|---|
| 1,024 | 7,744 | 4,780 |
| 4,096 | 28,612 | 6,376 |
| 8,192 | 30,787 | 3,605 |

For comparison, Brax PPO training of the same env on the CPU split into 16 JAX devices
(`--xla_force_host_platform_device_count=16`) ran at 21,600 steps/s including the learning updates — so
on this APU the GPU is a solid but not dramatic gain, and it competes with other GPU users for memory.
First compiles take 40–170 s per batch shape; a persistent compilation cache helps.

## `jaxcheck.py`: which set works, without hanging

```bash
python jaxcheck.py --label "my set"                        # devices, matmul, scatters, MJX reset/step
python jaxcheck.py --label "my set" --stages matmul scatter_set
XLA_FLAGS=--xla_… python jaxcheck.py --label "with a flag"
```

Each stage runs in its own process with its output streamed to `results/<label>-<stage>.log`, and is
recorded as **ok**, **fail** (non-zero exit; last error line kept), **fault** (a GPU fault signature was
printed — the process is killed at once, because after a queue abort ROCm processes can hang instead of
exiting) or **timeout**. Results go to `results/results.jsonl` and a table in `results/README.md`.
The MJX stages need `pip install playground`.

## Traps we hit (so you don't)

| Symptom | Cause | Fix |
|---|---|---|
| `Backend 'rocm' is not in the list of known backends`, `rocm_plugin_extension not found` | a 0.9.x plugin cannot find the ROCm libraries (no system ROCm) | use the 0.11.1 plugin from PyPI; for 0.9.x, `./rocm-run` sets the paths |
| `FAILED_PRECONDITION: No visible GPU devices`; `rocminfo`: *not member of "render" group* | `/dev/kfd` permission | `sudo usermod -aG render,video $USER`, log in again |
| a benchmark "compiles" for 25 minutes, one thread at 100 % | a GPU fault followed by a hang (0.9.x + MJX), not a compile | upgrade to JAX ≥ 0.10; use `jaxcheck.py` to see `fault` in seconds |
| `rocblas_gemm_ex failed with: rocblas_status_internal_error` in code that used to work | a persistent compilation cache shared between JAX versions ("PjRt-IFRT does not track XLA executable versions") | one cache per jaxlib/plugin build (`rocm-run` does this) |
| `Unknown flag in XLA_FLAGS: --xla_gpu_…` | the flag exists in the plugin's strings but jaxlib does not accept it | drop it; `jaxcheck.py` reports the line |
| out-of-memory next to a local LLM | the BIOS gives the iGPU a fixed carve-out (64 GiB here) shared by everything on the GPU | preallocation off; or set the BIOS UMA frame buffer to its minimum so the GPU uses GTT from the full RAM |

## Licence and disclaimer

Apache-2.0. Measurements from one machine on one day; your firmware, kernel and wheel versions may
differ — the matrix says exactly what was tested. Not affiliated with AMD or Google DeepMind.
