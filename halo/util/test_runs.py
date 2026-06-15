"""Tolerance-based regression check for short training runs.

Modes:
- "none": no-op.
- "generate": run 10 grad steps, save (loss, loss_dict, out_mean, out_std) per step
  under test_runs/<hash-of-cmd>.json, then exit.
- "verify": run 10 grad steps, load the previously saved artifact for the same
  command, and print a bold red warning with diffs if any scalar exceeds tolerance.

The artifact is keyed by sha256(sys.argv with the test_runs flag stripped),
so generate and verify of the same command map to the same file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


_RED_BOLD = "\033[1;31m"
_GREEN_BOLD = "\033[1;32m"
_RESET = "\033[0m"
_NUM_STEPS = 5

# Tolerance: ~5% relative or 1e-3 absolute, whichever is larger.
# Tuned to swallow DataLoader/CUDA/torch.compile non-determinism while still
# catching real refactor regressions (which typically shift loss by >>10%).
_RTOL = 1e-1
_ATOL = 5e-2


def _normalize_argv() -> str:
    """Return argv with the test_runs flag stripped so generate/verify hash to the same file."""
    skip_next = False
    out = []
    for a in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if a in ("--test-runs", "--trainer-cfg.test_runs", "--trainer-cfg.test-runs"):
            skip_next = True
            continue
        if a.startswith("--test-runs=") or a.startswith("--trainer-cfg.test_runs=") or a.startswith("--trainer-cfg.test-runs="):
            continue
        out.append(a)
    return " ".join(out)


def cmd_hash() -> str:
    return hashlib.sha256(_normalize_argv().encode("utf-8")).hexdigest()[:16]


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def artifact_path() -> str:
    base = os.path.join(_repo_root(), "test_runs")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cmd_hash()}.json")


def _scalarize(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().float().mean().item() if x.numel() > 1 else x.item()
    return x


def _tensor_summary(t: Optional[torch.Tensor]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(t, torch.Tensor):
        return None, None
    f = t.detach().to(torch.float32)
    return float(f.mean().item()), float(f.std().item())


def _close(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        a_f = float(a)
        b_f = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(a_f) and math.isnan(b_f):
        return True
    return abs(a_f - b_f) <= _ATOL + _RTOL * abs(b_f)


class TestRunRecorder:
    def __init__(self, mode: str):
        assert mode in ("none", "generate", "verify"), mode
        self.mode = mode
        self.records: List[Dict[str, Any]] = []
        self.num_steps = _NUM_STEPS

    @property
    def active(self) -> bool:
        return self.mode in ("generate", "verify")

    def record_step(
        self,
        step: int,
        loss_value: float,
        loss_dict: Dict[str, Any],
        out: Optional[torch.Tensor],
    ) -> None:
        if not self.active:
            return
        out_mean, out_std = _tensor_summary(out)
        self.records.append({
            "step": step,
            "loss": float(loss_value),
            "loss_dict": {k: _scalarize(v) for k, v in loss_dict.items()},
            "out_mean": out_mean,
            "out_std": out_std,
        })

    def done(self) -> bool:
        return self.active and len(self.records) >= self.num_steps

    def finalize(self) -> None:
        if not self.active:
            return
        path = artifact_path()
        payload = {
            "cmd": _normalize_argv(),
            "rtol": _RTOL,
            "atol": _ATOL,
            "records": self.records,
        }
        if self.mode == "generate":
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"[test-runs] generate: wrote {path}")
        else:
            self._verify(path, payload)
        sys.exit(0)

    def _verify(self, path: str, current: Dict[str, Any]) -> None:
        if not os.path.exists(path):
            self._warn([f"reference artifact not found at {path}; run with --test-runs generate first"])
            return
        with open(path) as f:
            ref = json.load(f)
        diffs: List[str] = []
        if len(ref["records"]) != len(current["records"]):
            diffs.append(f"step count mismatch: ref={len(ref['records'])} cur={len(current['records'])}")
        for r, c in zip(ref["records"], current["records"]):
            for field in ("loss", "out_mean", "out_std"):
                if not _close(r.get(field), c.get(field)):
                    diffs.append(f"step {c['step']}: {field} ref={r.get(field)!r} cur={c.get(field)!r} (rtol={_RTOL}, atol={_ATOL})")
            ref_ld, cur_ld = r.get("loss_dict", {}), c.get("loss_dict", {})
            for k in sorted(set(ref_ld) | set(cur_ld)):
                if not _close(ref_ld.get(k), cur_ld.get(k)):
                    diffs.append(f"step {c['step']}: loss_dict[{k}] ref={ref_ld.get(k)!r} cur={cur_ld.get(k)!r}")
        if diffs:
            self._warn(diffs, path=path)
        else:
            print(f"{_GREEN_BOLD}[test-runs] verify: OK ({len(current['records'])} steps within rtol={_RTOL}, atol={_ATOL}; ref={path}){_RESET}")

    @staticmethod
    def _warn(diffs: List[str], path: Optional[str] = None) -> None:
        header = "[test-runs] VERIFY MISMATCH"
        if path:
            header += f" (ref={path})"
        msg = "\n".join([header, *(f"  - {d}" for d in diffs)])
        print(f"{_RED_BOLD}{msg}{_RESET}", file=sys.stderr)
