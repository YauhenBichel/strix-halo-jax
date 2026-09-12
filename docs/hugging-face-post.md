# Hugging Face post

Written for huggingface.co/posts, which caps a post at about 2,000 characters. Kept here so the
numbers in it stay next to the evidence: every figure comes from docs/evidence-2026-09-11.md,
measured on 11 September 2026 on a Ryzen AI MAX+ 395.

---

**JAX and MuJoCo MJX now run on an AMD Ryzen AI MAX iGPU (Strix Halo, gfx1151) from pip wheels — no system ROCm install.**

If you searched for *"Backend 'rocm' is not in the list of known backends"*, *JAX ROCm gfx1151*, *MJX on AMD*, or *HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION*, this is the recipe that worked, plus a tool that tells you whether it works on **your** machine.

The hard part isn't installing — it's that a wrong wheel set doesn't error, it **hangs** or takes the GPU down with it. So `jaxcheck` runs each stage in its own process with a timeout and classifies it ok / fail / fault / timeout:

```
$ jaxcheck --label "jax-0.11.1 + rocm[libraries] 7.13.0"
devices        ok     3.0s
matmul         ok    15.0s
scatter_set    ok     3.5s
scatter_add    ok     4.0s
scatter_2d     ok     4.0s
scatter_vmap   ok     4.0s
scatter_oob    ok     4.0s
mjx_reset      ok    49.1s
mjx_step       ok   101.6s
```

Ryzen AI MAX+ 395 (Radeon 8060S), Ubuntu 24.04, kernel 7.0, **no system ROCm**, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, no other env vars. Seconds include compilation with no persistent cache. The GPU was simultaneously hosting a resident 52 GB LLM — these boxes have enough unified memory to do both.

The scatter stages are there because scatters are where ROCm builds tended to fall over on this architecture, and MJX leans on them heavily. Passing `matmul` tells you very little; passing `scatter_vmap` and `mjx_step` tells you the box can actually train a policy.

```
pip install strix-halo-jax
jaxcheck
```

It has no dependencies of its own — extras `gfx1151` and `mjx` install the tested wheel set. Apache-2.0. The compatibility matrix (JAX 0.11.1, 0.9.2 nightly, 0.9.1 release) and full evidence are in the repo.

If you have a Strix Halo box, I'd like your `jaxcheck` output — especially a failing one.

https://github.com/YauhenBichel/strix-halo-jax
