from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from halo.robomme_vqa import io as vqa_io
from halo.robomme_vqa.audit import (
    attach_visual_evidence,
    read_request_artifact,
    read_request_index,
)
from halo.robomme_vqa.io import write_json, write_jsonl
from halo.robomme_vqa.pipeline import VQARequest
from halo.robomme_vqa.timeline import TimelineFrame


def _request_row(
    tmp_path: Path, timesteps: tuple[int, ...] = (0, 2, 4)
) -> tuple[dict[str, object], VQARequest]:
    frames = []
    for timestep in timesteps:
        front = tmp_path / f"front-{timestep}.png"
        wrist = tmp_path / f"wrist-{timestep}.png"
        front.write_bytes(b"front")
        wrist.write_bytes(b"wrist")
        frames.append(
            TimelineFrame(
                timestep=timestep,
                front_image=str(front),
                wrist_image=str(wrist),
                event_boundary=timestep == 2,
                change_score=float(timestep),
            )
        )
    request = VQARequest(
        task_name="BinFill",
        suite_name="counting",
        episode_id="BinFill/episode_11",
        task_goal="put two cubes into the bin",
        query_timestep=4,
        timeline=tuple(frames),
        known_counts_by_family={"event_count": 1},
    )
    row: dict[str, object] = {
        "schema_version": "halo.robomme.vqa.request/v1",
        "task_name": request.task_name,
        "suite_name": request.suite_name,
        "episode_id": request.episode_id,
        "task_goal": request.task_goal,
        "query_timestep": request.query_timestep,
        "timeline": [
            {
                "timestep": frame.timestep,
                "front_image": frame.front_image,
                "wrist_image": frame.wrist_image,
                "event_boundary": frame.event_boundary,
                "change_score": frame.change_score,
            }
            for frame in request.timeline
        ],
        "known_counts_by_family": {"event_count": 1},
    }
    return row, request


def _record(request: VQARequest) -> dict[str, object]:
    return {
        "source_request_sha256": request.source_sha256(),
        "task_name": request.task_name,
        "suite_name": request.suite_name,
        "episode_id": request.episode_id,
        "query_timestep": request.query_timestep,
        "evidence_timestamps": [0, 2],
        "judge_result": {"evidence_timestamps": [2]},
    }


def _write_request_artifact(
    tmp_path: Path, row: dict[str, object], request: VQARequest
) -> tuple[Path, Path]:
    requests_path = tmp_path / "requests.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(requests_path, [row])
    images = []
    for frame in request.timeline:
        for camera, reference in (
            ("front", frame.front_image),
            ("wrist", frame.wrist_image),
        ):
            image_path = Path(reference)
            payload = image_path.read_bytes()
            images.append(
                {
                    "relative_path": str(image_path.relative_to(tmp_path)),
                    "encoded_sha256": hashlib.sha256(payload).hexdigest(),
                    "task_name": request.task_name,
                    "episode_id": request.episode_id,
                    "timestep": frame.timestep,
                    "camera": camera,
                }
            )
    write_json(
        manifest_path,
        {
            "schema_version": "halo.robomme.vqa.request-build-manifest/v1",
            "counts": {"requests": 1, "images": len(images)},
            "requests_jsonl": {
                "path": str(requests_path.resolve()),
                "sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
                "size_bytes": requests_path.stat().st_size,
                "request_schema_version": "halo.robomme.vqa.request/v1",
            },
            "images": images,
        },
    )
    return requests_path, manifest_path


def test_audit_hash_join_attaches_union_of_evidence_and_query(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    artifact = read_request_artifact(requests_path, manifest_path)

    joined = attach_visual_evidence(_record(request), artifact)
    visual = joined["visual_evidence"]
    assert [frame["timestep"] for frame in visual] == [0, 2, 4]
    assert visual[0]["candidate_evidence"] is True
    assert visual[1]["candidate_evidence"] is True
    assert visual[1]["judge_evidence"] is True
    assert visual[2]["query_frame"] is True
    assert all(Path(frame["front_image"]).is_file() for frame in visual)
    assert "visual_evidence" not in _record(request)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update(source_request_sha256="0" * 64), "no matching"),
        (lambda row: row.update(episode_id="BinFill/episode_12"), "episode_id mismatch"),
        (lambda row: row.update(query_timestep=2), "query_timestep mismatch"),
        (lambda row: row.update(evidence_timestamps=[3]), "unavailable"),
    ],
)
def test_audit_join_fails_closed_on_provenance_mismatch(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    artifact = read_request_artifact(requests_path, manifest_path)
    record = _record(request)
    mutation(record)
    with pytest.raises(ValueError, match=match):
        attach_visual_evidence(record, artifact)


def test_audit_join_rejects_duplicate_requests_and_missing_images(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path = tmp_path / "requests.jsonl"
    write_jsonl(requests_path, [row, row])
    with pytest.raises(ValueError, match="duplicate source_request_sha256"):
        read_request_index(requests_path)

    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    Path(request.timeline[0].front_image).unlink()
    with pytest.raises(FileNotFoundError):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_rejects_tampered_request_jsonl(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    requests_path.write_text(requests_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSONL SHA-256"):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_rejects_wrong_manifest_schema_and_path(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "halo.robomme.vqa.request-build-manifest/v2"
    write_json(manifest_path, payload)
    with pytest.raises(ValueError, match="schema_version"):
        read_request_artifact(requests_path, manifest_path)

    payload["schema_version"] = "halo.robomme.vqa.request-build-manifest/v1"
    payload["requests_jsonl"]["path"] = str(tmp_path / "substituted.jsonl")
    write_json(manifest_path, payload)
    with pytest.raises(ValueError, match="path mismatch"):
        read_request_artifact(requests_path, manifest_path)


def test_audit_join_rejects_image_content_tampering(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    Path(request.timeline[0].front_image).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="front image at timestep 0 SHA-256"):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_binds_timeline_images_the_audit_never_selects(
    tmp_path: Path,
) -> None:
    # Timestep 1 is in the request timeline but outside the record's evidence,
    # so only a whole-timeline check at load time can catch tampering with it.
    row, request = _request_row(tmp_path, timesteps=(0, 1, 2, 4))
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    unaudited = Path(request.timeline[1].front_image)
    assert 1 not in set(_record(request)["evidence_timestamps"]) | {
        _record(request)["query_timestep"]
    }

    unaudited.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="front image at timestep 1 SHA-256"):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_rejects_timeline_images_missing_from_the_manifest(
    tmp_path: Path,
) -> None:
    row, request = _request_row(tmp_path, timesteps=(0, 1, 2, 4))
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["images"] = [
        image for image in payload["images"] if image["timestep"] != 1
    ]
    payload["counts"]["images"] = len(payload["images"])
    write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="timestep 1 is absent from request build manifest"):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_rejects_timeline_image_identity_drift(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for image in payload["images"]:
        if image["timestep"] == 0 and image["camera"] == "front":
            image["episode_id"] = "BinFill/episode_99"
    write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="metadata does not match its source request"):
        read_request_artifact(requests_path, manifest_path)


def test_request_artifact_hashes_exactly_the_bytes_it_parsed(tmp_path: Path) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    original_requests = requests_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    artifact = read_request_artifact(requests_path, manifest_path)

    # A later write must not change the provenance of what was already parsed.
    requests_path.write_bytes(original_requests + b"\n")
    manifest_path.write_bytes(original_manifest + b"\n")

    assert artifact.requests_sha256 == hashlib.sha256(original_requests).hexdigest()
    assert artifact.requests_size_bytes == len(original_requests)
    assert artifact.manifest_sha256 == hashlib.sha256(original_manifest).hexdigest()
    assert artifact.manifest_size_bytes == len(original_manifest)


def _full_record(request: VQARequest) -> dict[str, object]:
    """A complete v1 corpus record consistent with ``request``."""
    return {
        "schema_version": "halo.robomme.vqa.record/v1",
        "task_name": request.task_name,
        "suite_name": request.suite_name,
        "episode_id": request.episode_id,
        "query_timestep": request.query_timestep,
        "candidate_id": "candidate-1",
        "question": "How many cubes were already in the bin?",
        "answer": "two",
        "question_family": "event_count",
        "evidence_timestamps": [0, 2],
        "generator_model_revision": "revision-a",
        "generator_prompt_revision": "generator-v1",
        "judge_model_revision": "revision-a",
        "judge_prompt_revision": "judge-v1",
        "source_request_sha256": request.source_sha256(),
        "generator_response_sha256": "1" * 64,
        "judge_response_sha256": "2" * 64,
        "deterministic_checks": {
            "identifiers_valid": True,
            "timestamps_valid": True,
            "no_privileged_terms": True,
            "count_valid": True,
            "not_duplicate": True,
            "answer_within_bounds": True,
        },
        "judge_result": {
            "schema_version": "halo.robomme.vqa.judge/v1",
            "visually_answerable": True,
            "history_required": True,
            "answer_correct": True,
            "unambiguous": True,
            "future_leakage": False,
            "evidence_timestamps": [2],
            "reason_code": "accepted",
        },
        "accepted": True,
        "rejection_reason": None,
    }


def _audit_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_robomme_vqa_toctou",
        Path(__file__).resolve().parents[1] / "scripts/vqa/audit_robomme_vqa.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_manifest_hashes_survive_a_source_rewrite_during_packet_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, request = _request_row(tmp_path)
    requests_path, manifest_path = _write_request_artifact(tmp_path, row, request)
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records_path, [_full_record(request)])
    sources = (records_path, requests_path, manifest_path)
    original = {path: path.read_bytes() for path in sources}

    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    cli = _audit_cli()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    # A concurrent writer touches every source after parsing but before the
    # audit manifest records their hashes.
    real_attach = cli.attach_visual_evidence

    def attach_then_rewrite(record: Any, artifact: Any) -> Any:
        result = real_attach(record, artifact)
        for path in sources:
            path.write_bytes(original[path] + b"\n")
        return result

    monkeypatch.setattr(cli, "attach_visual_evidence", attach_then_rewrite)
    output_dir = approved_root / "audit-001"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_robomme_vqa.py",
            "--records-jsonl", str(records_path),
            "--requests-jsonl", str(requests_path),
            "--request-manifest", str(manifest_path),
            "--output-dir", str(output_dir),
            "--stage", "pilot",
        ],
    )
    cli.main()

    audit_manifest = json.loads((output_dir / "audit_manifest.json").read_text())
    assert audit_manifest["sample_count"] == 1
    for field, path in (
        ("source_records_sha256", records_path),
        ("source_requests_sha256", requests_path),
        ("source_request_manifest_sha256", manifest_path),
    ):
        assert audit_manifest[field] == hashlib.sha256(original[path]).hexdigest()
        assert audit_manifest[field] != hashlib.sha256(path.read_bytes()).hexdigest()
