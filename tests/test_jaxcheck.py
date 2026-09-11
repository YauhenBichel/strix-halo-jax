"""The classification that makes jaxcheck hang-safe, on CPU with scripted stages."""

import pytest

import jaxcheck


@pytest.fixture(autouse=True)
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(jaxcheck, "OUT", tmp_path)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    return tmp_path


def test_every_stage_is_valid_python():
    for s in jaxcheck.STAGES:
        compile(jaxcheck.stage_code(s), s, "exec")


def test_a_real_stage_passes_on_cpu(out):
    row = jaxcheck.run_stage("scatter_oob", "t", 120)
    assert row["status"] == "ok" and row["exit_code"] == 0
    assert (out / "t-scatter_oob.log").read_text().startswith("OK")


def test_a_crash_is_fail_with_its_error():
    row = jaxcheck.run_stage("matmul", "t", 30, code="raise RuntimeError('bad shape')")
    assert row["status"] == "fail" and "bad shape" in row["error"]


def test_a_gpu_fault_then_a_hang_is_stopped_early_as_fault():
    code = ("import sys, time\nprint('Callback: Queue aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION',"
            " file=sys.stderr, flush=True)\ntime.sleep(60)")
    row = jaxcheck.run_stage("mjx_reset", "t", 30, code=code)
    assert row["status"] == "fault" and row["seconds"] < 10 and "APERTURE" in row["error"]


def test_a_silent_hang_is_timeout():
    row = jaxcheck.run_stage("matmul", "t", 2, code="import time; time.sleep(60)")
    assert row["status"] == "timeout" and row["exit_code"] is None


def test_an_unknown_xla_flag_is_reported():
    code = "import sys; print('parse_flags_from_env.cc:234] Unknown flag in XLA_FLAGS: --x', file=sys.stderr); sys.exit(134)"
    row = jaxcheck.run_stage("matmul", "t", 30, xla_flags="--x", code=code)
    assert row["status"] == "fail" and "Unknown flag" in row["error"] and row["xla_flags"] == "--x"


def test_render_escapes_pipes():
    md = jaxcheck.render([{"time": "2026-09-11T17:00:00+0100", "label": "a", "stage": "matmul", "status": "fail",
                           "seconds": 1.0, "xla_flags": "", "error": "a|b"}])
    assert "| a/b |" in md
