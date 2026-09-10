"""End-to-end tests for the two-process loopback runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_loopback_network.py"


def _final_report(stdout: str) -> dict:
    reports = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    assert reports, stdout
    return reports[-1]


def test_two_process_runner_mines_and_reconciles(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--state-dir",
            str(tmp_path),
            "--run-for",
            "2",
            "--mine-interval",
            "0.2",
            "--max-nonces",
            "128",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = _final_report(result.stdout)
    assert report["loopback_only"] is True
    assert report["processes_started"] == 2
    assert report["alice"]["height"] >= 1
    assert report["bob"]["height"] >= 1
    assert report["alice"]["tip"] == report["bob"]["tip"]
    assert report["alice"]["sync"]["successes"] >= 1 or report["bob"]["sync"]["successes"] >= 1


def test_two_process_runner_rejects_non_loopback_peer(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--state-dir",
            str(tmp_path),
            "--peer",
            "http://example.invalid:1234",
            "--run-for",
            "0.1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "loopback" in (result.stdout + result.stderr).lower()
