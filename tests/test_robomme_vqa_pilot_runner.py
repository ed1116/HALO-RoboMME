from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_robomme_vqa_pilot_cli",
        REPOSITORY_ROOT / "scripts/vqa/run_robomme_vqa_pilot.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(source_hash: str, candidate: int) -> dict[str, Any]:
    return {
        "source_request_sha256": source_hash,
        "candidate_id": f"candidate-{candidate}",
        "accepted": True,
    }


def _progress(source_hash: str, index: int, count: int) -> dict[str, Any]:
    return {
        "schema_version": "halo.robomme.vqa.pilot-progress/v1",
        "request_index": index,
        "source_request_sha256": source_hash,
        "status": "ok",
        "record_count": count,
    }


def test_partial_trailing_line_from_a_crash_is_discarded(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "records.jsonl"
    runner.append_jsonl(path, [_record("a" * 64, 1), _record("a" * 64, 2)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"source_request_sha256": "bbb')  # interrupted mid-write

    rows = runner.read_whole_lines(path)
    assert [row["candidate_id"] for row in rows] == ["candidate-1", "candidate-2"]


def test_records_without_a_progress_line_are_dropped_so_work_is_redone(
    tmp_path: Path,
) -> None:
    """Progress is appended after records, so records may lead progress on a crash."""
    runner = _runner()
    records_path = tmp_path / "records.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    done, interrupted = "a" * 64, "b" * 64

    runner.append_jsonl(records_path, [_record(done, 1), _record(done, 2)])
    runner.append_jsonl(progress_path, [_progress(done, 0, 2)])
    # Crash: the next request's records landed, its progress line did not.
    runner.append_jsonl(records_path, [_record(interrupted, 1)])

    records, progress = runner.recover_completed_work(records_path, progress_path)
    assert [row["source_request_sha256"] for row in records] == [done, done]
    assert [row["source_request_sha256"] for row in progress] == [done]
    # The truncation is persisted, so a resumed run cannot double-count.
    assert len(runner.read_whole_lines(records_path)) == 2
    assert interrupted not in records_path.read_text(encoding="utf-8")


def test_fully_recorded_requests_survive_recovery_untouched(tmp_path: Path) -> None:
    runner = _runner()
    records_path = tmp_path / "records.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    first, second = "a" * 64, "b" * 64
    runner.append_jsonl(records_path, [_record(first, 1)])
    runner.append_jsonl(progress_path, [_progress(first, 0, 1)])
    runner.append_jsonl(records_path, [_record(second, 1), _record(second, 2)])
    runner.append_jsonl(progress_path, [_progress(second, 1, 2)])
    before = records_path.read_bytes()

    records, progress = runner.recover_completed_work(records_path, progress_path)
    assert len(records) == 3
    assert len(progress) == 2
    assert records_path.read_bytes() == before


def test_recovery_of_an_untouched_directory_is_empty(tmp_path: Path) -> None:
    runner = _runner()
    records, progress = runner.recover_completed_work(
        tmp_path / "records.jsonl", tmp_path / "progress.jsonl"
    )
    assert records == [] and progress == []


def test_rewrite_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "records.jsonl"
    runner.rewrite_jsonl(path, [_record("a" * 64, 1)])
    assert [entry.name for entry in tmp_path.iterdir()] == ["records.jsonl"]
    assert len(runner.read_whole_lines(path)) == 1


def test_output_directory_guard_distinguishes_fresh_resumable_and_finished(
    tmp_path: Path,
) -> None:
    runner = _runner()
    fresh = tmp_path / "pilot-fresh"
    runner.prepare_output_dir(fresh, resume=False)
    assert fresh.is_dir()

    with pytest.raises(FileExistsError):
        runner.prepare_output_dir(fresh, resume=False)

    # An unfinished run has no manifest, so it may be resumed.
    runner.prepare_output_dir(fresh, resume=True)

    (fresh / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="completed pilot"):
        runner.prepare_output_dir(fresh, resume=True)
