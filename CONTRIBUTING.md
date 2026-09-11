# Contributing

The most useful contribution is **a row in the compatibility matrix**:

1. Install a wheel set (write down every version and index), run
   `python jaxcheck.py --label "<short name of the set>"`.
2. Open a pull request adding a row to the README's matrix, with `results/README.md` pasted in the
   description, your GPU (`rocminfo | grep gfx`), kernel and whether a system ROCm is installed.
3. Failures are as welcome as successes: a `fault` or `fail` row saves the next person a day.

Code changes: one change per pull request, with a test (`uv run python -m pytest`; the tests run on
CPU with scripted stages).
