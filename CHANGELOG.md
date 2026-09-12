# Changelog

## 0.1.1

- The demo gif is no longer in the source distribution: 1.04 MB -> 14 kB. The README points at it by
  URL, which also makes it render on the PyPI page, where relative links cannot resolve.

## 0.1.0 (2026-09-12)

- The working recipe for JAX 0.11.1 and MuJoCo MJX on gfx1151 from pip wheels, no system ROCm, and
  the compatibility matrix (JAX 0.11.1, 0.9.2 nightly, 0.9.1 release).
- `jaxcheck`: staged, hang-safe checks (devices, matmul, five scatters, MJX reset and step), each in
  its own process, classified ok / fail / fault / timeout. Installable from PyPI as a command with no
  dependencies of its own; extras `gfx1151` and `mjx` install the tested set.
- `rocm-run` for the 0.9.x wheel sets: library paths and one compilation cache per build.
