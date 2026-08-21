from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from halo.robomme import TASKS, write_manifest
from halo.robomme_vqa.audit import read_request_artifact
from halo.robomme_vqa import io as vqa_io
from halo.robomme_vqa import request_builder
from halo.robomme_vqa.io import read_request_jsonl
from halo.robomme_vqa.prompts import TASK_INSTRUCTIONS, multimodal_messages
from halo.robomme_vqa.request_builder import (
    BUILD_MANIFEST_SCHEMA_VERSION,
    build_pilot_requests,
    plan_pilot_requests,
)
from robomme_hdf5_fixture import write_fixture


def _dev_keys(task: str, episodes: int, seed: int = 0) -> set[str]:
    ranked = sorted(
        (f"episode_{index}" for index in range(episodes)),
        key=lambda key: hashlib.sha256(f"{seed}:{task}:{key}".encode()).digest(),
    )
    return set(ranked[:10])


def test_builder_uses_train_split_is_causal_and_hashes_only_selected_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_fixture(raw_root, episodes=12, timesteps=6)
    manifest_path = tmp_path / "shared-manifest.json"
    write_manifest(raw_root, manifest_path, validate_demo_prefix=True)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    first_plan = plan_pilot_requests(
        raw_root, manifest_path, target_requests=16, max_evidence_frames=4
    )
    second_plan = plan_pilot_requests(
        raw_root, manifest_path, target_requests=16, max_evidence_frames=4
    )
    assert first_plan == second_plan

    output = build_pilot_requests(
        raw_root,
        manifest_path,
        approved_root / "pilot-001",
        target_requests=16,
        max_evidence_frames=4,
    )
    requests = tuple(read_request_jsonl(output / "requests.jsonl"))
    manifest = json.loads((output / "manifest.json").read_text())
    audit_source = read_request_artifact(
        output / "requests.jsonl", output / "manifest.json"
    )

    assert manifest["schema_version"] == BUILD_MANIFEST_SCHEMA_VERSION
    assert len(requests) == 16
    assert len(audit_source.requests) == 16
    assert {request.task_name for request in requests} == set(TASKS)
    assert set(manifest["counts"]["eligible_requests_by_question_family"]) == {
        family
        for _, families in TASK_INSTRUCTIONS.values()
        for family in families
    }
    assert manifest["split"] == {
        "name": "train",
        "split_seed": 0,
        "dev_episodes_per_task": 10,
        "unit": "whole_episode",
        "selection": "canonical RoboMMEHDF5Dataset stable-hash split",
    }
    assert all(request.known_counts_by_family == {} for request in requests)
    assert manifest["selection"]["privileged_annotation_use"] == (
        "frame selection only; values are not included in visual prompts"
    )
    for request in requests:
        episode_key = request.episode_id.split("/", 1)[1]
        assert episode_key not in _dev_keys(request.task_name, 12)
        assert request.timeline[-1].timestep == request.query_timestep
        assert all(frame.timestep <= request.query_timestep for frame in request.timeline)
        assert any(frame.timestep < request.query_timestep for frame in request.timeline)
        content = multimodal_messages("visual-only audit", request.timeline)[0]["content"]
        prompt_text = "\n".join(
            item["text"] for item in content if item["type"] == "text"
        )
        assert "subgoal" not in prompt_text
        for frame in request.timeline:
            assert Path(frame.front_image).is_relative_to(output / "images")
            assert Path(frame.wrist_image).is_relative_to(output / "images")

    pngs = sorted((output / "images").rglob("*.png"))
    expected_images = 2 * sum(len(request.timeline) for request in requests)
    assert len(pngs) == manifest["counts"]["images"] == expected_images
    assert len(manifest["images"]) == expected_images
    assert len(audit_source.images) == expected_images
    for image in manifest["images"]:
        payload = (output / image["relative_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == image["encoded_sha256"]
    assert "breaking v2" in manifest["request_v1_provenance_scope"]["future_schema_note"]


def test_builder_fails_before_writing_for_stale_manifest_or_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_fixture(raw_root, episodes=12, timesteps=4)
    manifest_path = tmp_path / "shared-manifest.json"
    write_manifest(raw_root, manifest_path)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    existing = approved_root / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        build_pilot_requests(
            raw_root, manifest_path, existing, target_requests=16, max_evidence_frames=2
        )

    with (raw_root / "record_dataset_BinFill.h5").open("ab") as handle:
        handle.write(b"stale")
    fresh_output = approved_root / "must-not-exist"
    with pytest.raises(ValueError, match="stale manifest"):
        build_pilot_requests(
            raw_root,
            manifest_path,
            fresh_output,
            target_requests=16,
            max_evidence_frames=2,
        )
    assert not fresh_output.exists()


def test_default_pilot_allocation_is_bounded_and_task_balanced() -> None:
    from halo.robomme_vqa.request_builder import _request_counts_by_task

    counts = _request_counts_by_task(100)
    assert sum(counts.values()) == 100
    assert set(counts) == set(TASKS)
    assert max(counts.values()) - min(counts.values()) == 1
    with pytest.raises(ValueError, match="at least 16"):
        _request_counts_by_task(15)


def test_builder_rejects_manifest_episode_keys_that_escape_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_fixture(raw_root, episodes=12, timesteps=4)
    manifest_path = tmp_path / "shared-manifest.json"
    write_manifest(raw_root, manifest_path)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    # "/episode_N" addresses the same HDF5 group and passes the dataset's
    # numeric-suffix check, but it is absolute, so joining it onto the artifact
    # path would discard the artifact root and write at the filesystem root.
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for episode in payload["tasks"]["BinFill"]["episodes"]:
        episode["episode_key"] = f"/{episode['episode_key']}"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert Path("images/BinFill") / "/episode_3" == Path("/episode_3")

    output = approved_root / "must-not-exist"
    with pytest.raises(ValueError, match="manifest episode key is not episode_N"):
        build_pilot_requests(
            raw_root, manifest_path, output, target_requests=16, max_evidence_frames=2
        )
    assert not output.exists()
    assert list(approved_root.iterdir()) == []


def test_builder_leaves_no_artifact_or_staging_directory_when_a_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_fixture(raw_root, episodes=12, timesteps=4)
    manifest_path = tmp_path / "shared-manifest.json"
    write_manifest(raw_root, manifest_path)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    real_save_png = request_builder._save_png
    calls: list[Path] = []

    def failing_save_png(path: Path, value: object) -> dict[str, object]:
        calls.append(path)
        if len(calls) == 5:
            raise OSError("disk full")
        return real_save_png(path, value)

    monkeypatch.setattr(request_builder, "_save_png", failing_save_png)
    output = approved_root / "pilot-001"
    with pytest.raises(OSError, match="disk full"):
        build_pilot_requests(
            raw_root, manifest_path, output, target_requests=16, max_evidence_frames=2
        )
    assert not output.exists()
    assert list(approved_root.iterdir()) == []

    # The interrupted attempt must not block a clean retry.
    monkeypatch.setattr(request_builder, "_save_png", real_save_png)
    assert build_pilot_requests(
        raw_root, manifest_path, output, target_requests=16, max_evidence_frames=2
    ) == output
    assert (output / "requests.jsonl").is_file()


def test_completed_artifact_uses_final_paths_and_verifies_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_fixture(raw_root, episodes=12, timesteps=4)
    manifest_path = tmp_path / "shared-manifest.json"
    write_manifest(raw_root, manifest_path)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir()
    monkeypatch.setattr(vqa_io, "VQA_OUTPUT_ROOT", approved_root)

    output = build_pilot_requests(
        raw_root, manifest_path, approved_root / "pilot-001",
        target_requests=16, max_evidence_frames=2,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["requests_jsonl"]["path"] == str(output / "requests.jsonl")
    assert manifest["requests_jsonl"]["sha256"] == hashlib.sha256(
        (output / "requests.jsonl").read_bytes()
    ).hexdigest()
    assert "incomplete" not in json.dumps(manifest)
    assert list(approved_root.iterdir()) == [output]
    # read_request_artifact re-verifies every declared path against the artifact.
    read_request_artifact(output / "requests.jsonl", output / "manifest.json")


@pytest.mark.parametrize(
    ("execution_start", "boundaries", "expected_candidates"),
    [
        # A boundary at index t yields the frame after the event, t + 1.
        # No video-demo prefix: the timestep-0 boundary would map to timestep 1,
        # a memory question with one earlier frame. It must be dropped.
        (0, (0, 60, 120), (61, 121)),
        # Video-demo prefix: the timestep-0 boundary maps to the first frame
        # after the demo, which carries the whole demonstration as history.
        (66, (0, 90, 150), (66, 91, 151)),
    ],
)
def test_query_floor_keeps_post_demo_frames_and_drops_one_frame_history(
    execution_start: int,
    boundaries: tuple[int, ...],
    expected_candidates: tuple[int, ...],
) -> None:
    from halo.robomme_vqa.request_builder import SourceEpisode, _query_timestep

    episode = SourceEpisode(
        task_name="BinFill",
        suite_name="counting",
        file_path=Path("/nonexistent.h5"),
        episode_key="episode_0",
        episode_id=0,
        num_timesteps=200,
        task_goal="goal",
        execution_start=execution_start,
    )
    flags = [index in boundaries for index in range(episode.num_timesteps)]

    # Vary the seed to enumerate which candidates the stable rank can reach.
    reachable = {
        _query_timestep(episode, flags, seed=seed, min_history=16)
        for seed in range(200)
    }
    assert reachable == set(expected_candidates)
    assert all(query >= 16 for query in reachable)


def test_query_floor_falls_back_when_no_candidate_has_enough_history() -> None:
    from halo.robomme_vqa.request_builder import SourceEpisode, _query_timestep

    episode = SourceEpisode(
        task_name="BinFill",
        suite_name="counting",
        file_path=Path("/nonexistent.h5"),
        episode_key="episode_0",
        episode_id=0,
        num_timesteps=6,
        task_goal="goal",
        execution_start=0,
    )
    flags = [index == 0 for index in range(episode.num_timesteps)]
    # A short fixture episode cannot satisfy the floor; selection must still work.
    assert _query_timestep(episode, flags, seed=0, min_history=16) == 1


def test_default_pilot_allocation_gives_every_task_the_same_count() -> None:
    from halo.robomme_vqa.request_builder import _request_counts_by_task

    counts = _request_counts_by_task(96)
    assert set(counts) == set(TASKS)
    assert set(counts.values()) == {6}
    assert sum(counts.values()) == 96
