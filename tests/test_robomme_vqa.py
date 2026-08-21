from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

import halo.robomme_vqa.contracts as contracts
from halo.robomme.dataset import TASKS, TASK_TO_SUITE
from halo.robomme_vqa.contracts import (
    APPROVED_REVISION,
    Candidate,
    ContractError,
    JudgeResult,
    extract_json_object,
    load_model_config,
    parse_candidates,
    parse_judge_result,
    parse_record,
)
from halo.robomme_vqa.io import guarded_output_directory, request_from_dict
from halo.robomme_vqa.pipeline import (
    CandidateGenerator,
    IndependentJudge,
    OfflineVQAPipeline,
    VQARequest,
)
from halo.robomme_vqa.prompts import TASK_INSTRUCTIONS, multimodal_messages
from halo.robomme_vqa.timeline import TimelineFrame, select_causal_evidence
from halo.robomme_vqa.validation import (
    EpisodeDeduplicator,
    ValidationContext,
    rejection_reason,
    validate_candidate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPOSITORY_ROOT / "config/vqa/robomme_qwen3_vl_8b_v1.json"


class FakeBackend:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        generation: Mapping[str, Any],
    ) -> str:
        self.calls.append((messages, generation))
        return self.outputs.pop(0)


def candidate_response(
    *,
    question: str = "How many placements were completed before timestep 12?",
    answer: str = "two placements",
    family: str = "event_count",
    timestamps: list[int] | None = None,
) -> str:
    return json.dumps(
        {
            "schema_version": "halo.robomme.vqa.candidates/v1",
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "question_family": family,
                    "question": question,
                    "answer": answer,
                    "evidence_timestamps": timestamps or [4, 8],
                }
            ],
        }
    )


def judge_response(
    *,
    timestamps: list[int] | None = None,
    answer_correct: bool = True,
    reason_code: str = "accepted",
) -> str:
    return json.dumps(
        {
            "schema_version": "halo.robomme.vqa.judge/v1",
            "visually_answerable": True,
            "history_required": True,
            "answer_correct": answer_correct,
            "unambiguous": True,
            "future_leakage": False,
            "evidence_timestamps": timestamps or [4, 8],
            "reason_code": reason_code,
        }
    )


def timeline() -> tuple[TimelineFrame, ...]:
    return tuple(
        TimelineFrame(
            timestep=timestep,
            front_image=f"/data/front-{timestep}.png",
            wrist_image=f"/data/wrist-{timestep}.png",
            event_boundary=timestep in {4, 8},
            change_score=float(timestep),
        )
        for timestep in (0, 4, 8, 12, 20)
    )


def test_json_extraction_is_tolerant_of_fences_but_strict_about_contract() -> None:
    raw = f"analysis is not shared\n```json\n{candidate_response()}\n```"
    assert extract_json_object(raw, required_key="candidates")["schema_version"].endswith("/v1")
    assert parse_candidates(raw)[0].candidate_id == "candidate-1"

    with pytest.raises(ContractError, match="exactly one"):
        parse_candidates(f"{candidate_response()}\n{candidate_response(answer='three')}")
    with pytest.raises(ContractError, match="fields differ"):
        parse_candidates(candidate_response()[:-1] + ', "extra": true}')
    with pytest.raises(ContractError):
        parse_judge_result(judge_response().replace('"accepted"', '"Accepted"'))


def test_pinned_config_and_schemas_are_strict_json() -> None:
    config = load_model_config(MODEL_CONFIG)
    assert config.model_revision == APPROVED_REVISION
    assert config.processor_revision == APPROVED_REVISION
    assert config.dtype == "float16"
    assert config.attention_implementation == "sdpa"
    assert config.torch_version == "2.9.1+cu128"
    assert config.transformers_version == "5.15.1"
    assert config.qwen_vl_utils_version == "0.0.14"
    assert config.max_images_per_call == 32
    assert config.max_video_frames == 0
    assert config.judge_generation == {
        "do_sample": False,
        "max_new_tokens": 512,
        "temperature": 0.0,
    }
    for path in sorted((REPOSITORY_ROOT / "schemas/vqa/v1").glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False


def test_runtime_contract_fails_closed_on_package_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_model_config(MODEL_CONFIG)
    expected = {
        "torch": "2.9.1+cu128",
        "transformers": "5.15.1",
        "qwen-vl-utils": "0.0.14",
    }
    monkeypatch.setattr(contracts, "version", expected.__getitem__)
    assert config.verify_runtime() == expected
    monkeypatch.setattr(
        contracts,
        "version",
        lambda package: "0.0.0" if package == "transformers" else expected[package],
    )
    with pytest.raises(RuntimeError, match="transformers runtime mismatch"):
        config.verify_runtime()


def test_all_tasks_have_suite_and_question_family_prompts() -> None:
    assert set(TASK_INSTRUCTIONS) == set(TASKS)
    assert len(TASK_INSTRUCTIONS) == 16
    for task, (_, families) in TASK_INSTRUCTIONS.items():
        assert TASK_TO_SUITE[task] in {"counting", "permanence", "reference", "imitation"}
        assert families


def test_evidence_selection_is_deterministic_and_never_uses_future() -> None:
    selected = select_causal_evidence(timeline(), query_timestep=12, max_frames=3)
    assert [frame.timestep for frame in selected] == [4, 8, 12]
    assert max(frame.timestep for frame in selected) <= 12
    assert select_causal_evidence(timeline(), query_timestep=12, max_frames=3) == selected
    content = multimodal_messages("prompt", selected)[0]["content"]
    labels = [item["text"] for item in content if item["type"] == "text"]
    assert any("camera=front" in label for label in labels)
    assert any("camera=wrist" in label for label in labels)


def test_generator_and_judge_are_fresh_calls_without_reasoning_sharing() -> None:
    config = load_model_config(MODEL_CONFIG)
    generator_backend = FakeBackend(
        [f"SECRET_GENERATOR_REASONING\n```json\n{candidate_response()}\n```"]
    )
    judge_backend = FakeBackend([judge_response()])
    request = VQARequest(
        task_name="BinFill",
        suite_name="counting",
        episode_id="BinFill/episode_0",
        task_goal="place two blue cubes into the bin",
        query_timestep=12,
        timeline=timeline(),
        known_counts_by_family={"event_count": 2},
    )
    pipeline = OfflineVQAPipeline(
        CandidateGenerator(generator_backend, config),
        IndependentJudge(judge_backend, config),
        max_evidence_frames=3,
    )
    records = pipeline.process(request, candidate_count=1)

    assert len(generator_backend.calls) == len(judge_backend.calls) == 1
    generator_prompt = generator_backend.calls[0][0][0]["content"][0]["text"]
    judge_prompt = judge_backend.calls[0][0][0]["content"][0]["text"]
    assert "construct concise visual-memory" in generator_prompt
    assert "independent deterministic" in judge_prompt
    assert "SECRET_GENERATOR_REASONING" not in judge_prompt
    assert generator_prompt != judge_prompt
    assert records[0]["accepted"] is True
    assert records[0]["generator_model_revision"] == APPROVED_REVISION
    assert records[0]["source_request_sha256"] == request.source_sha256()
    record_schema = json.loads(
        (REPOSITORY_ROOT / "schemas/vqa/v1/record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(records[0]) == set(record_schema["required"])


@pytest.mark.parametrize(
    ("candidate", "judge", "known_count", "expected_reason"),
    [
        (
            Candidate(
                "candidate-1",
                "event_count",
                "How many events had completed by frame 99?",
                "two",
                (0, 8),
            ),
            JudgeResult(True, True, True, True, False, (0, 8), "accepted"),
            2,
            "deterministic_timestamps_valid",
        ),
        (
            Candidate(
                "candidate-1",
                "event_count",
                "What does is_subgoal_boundary say?",
                "two",
                (0, 8),
            ),
            JudgeResult(True, True, True, True, False, (0, 8), "accepted"),
            2,
            "deterministic_no_privileged_terms",
        ),
        (
            Candidate(
                "candidate-1",
                "event_count",
                "How many placements were completed?",
                "three",
                (0, 8),
            ),
            JudgeResult(True, True, True, True, False, (0, 8), "accepted"),
            2,
            "deterministic_count_valid",
        ),
    ],
)
def test_deterministic_leakage_privilege_and_count_checks(
    candidate: Candidate,
    judge: JudgeResult,
    known_count: int,
    expected_reason: str,
) -> None:
    context = ValidationContext(
        task_name="BinFill",
        suite_name="counting",
        episode_id="BinFill/episode_0",
        query_timestep=12,
        permitted_timestamps=(0, 4, 8, 12),
    )
    checks = validate_candidate(
        candidate,
        judge,
        context,
        deduplicator=EpisodeDeduplicator(),
        known_count=known_count,
    )
    assert rejection_reason(checks, judge) == expected_reason


def test_near_duplicates_are_rejected_only_within_episode() -> None:
    candidate = Candidate(
        "candidate-1",
        "event_count",
        "How many blue cube placements were completed earlier?",
        "two",
        (0, 8),
    )
    similar = Candidate(
        "candidate-1",
        "event_count",
        "Earlier, how many blue cube placements were completed?",
        "two",
        (0, 8),
    )
    judge = JudgeResult(True, True, True, True, False, (0, 8), "accepted")
    deduplicator = EpisodeDeduplicator(jaccard_threshold=0.8)
    context = ValidationContext("BinFill", "counting", "BinFill/episode_0", 12, (0, 8, 12))
    assert validate_candidate(
        candidate, judge, context, deduplicator=deduplicator, known_count=2
    ).not_duplicate
    assert not validate_candidate(
        similar, judge, context, deduplicator=deduplicator, known_count=2
    ).not_duplicate
    other_episode = ValidationContext(
        "BinFill", "counting", "BinFill/episode_1", 12, (0, 8, 12)
    )
    assert validate_candidate(
        similar, judge, other_episode, deduplicator=deduplicator, known_count=2
    ).not_duplicate


def test_request_contract_and_output_guard() -> None:
    payload = {
        "schema_version": "halo.robomme.vqa.request/v1",
        "task_name": "MoveCube",
        "suite_name": "imitation",
        "episode_id": "MoveCube/episode_3",
        "task_goal": "repeat the demonstrated strategy",
        "query_timestep": 4,
        "timeline": [
            {
                "timestep": 0,
                "front_image": "/data/front.png",
                "wrist_image": "/data/wrist.png",
                "event_boundary": True,
                "change_score": 1.0,
            }
        ],
        "known_counts_by_family": {},
    }
    assert request_from_dict(payload).task_name == "MoveCube"
    assert guarded_output_directory(
        "/data/ed1116/robomme/vqa/halo/pilot-001"
    ) == Path("/data/ed1116/robomme/vqa/halo/pilot-001")
    with pytest.raises(ValueError):
        guarded_output_directory("relative-output")
    with pytest.raises(ValueError):
        guarded_output_directory("/tmp/not-approved")


def _valid_record() -> dict[str, Any]:
    config = load_model_config(MODEL_CONFIG)
    pipeline = OfflineVQAPipeline(
        CandidateGenerator(FakeBackend([candidate_response()]), config),
        IndependentJudge(FakeBackend([judge_response()]), config),
        max_evidence_frames=3,
    )
    request = VQARequest(
        task_name="BinFill",
        suite_name="counting",
        episode_id="BinFill/episode_0",
        task_goal="place two blue cubes into the bin",
        query_timestep=12,
        timeline=timeline(),
        known_counts_by_family={"event_count": 2},
    )
    return pipeline.process(request, candidate_count=1)[0]


def test_record_fields_match_the_published_record_schema() -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "schemas/vqa/v1/record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert contracts.RECORD_FIELDS == set(schema["required"])
    assert contracts.DETERMINISTIC_CHECK_FIELDS == set(
        schema["properties"]["deterministic_checks"]["required"]
    )
    assert parse_record(_valid_record()) == _valid_record()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"accepted": "false"}, "accepted must be boolean"),
        ({"accepted": 0}, "accepted must be boolean"),
        ({"query_timestep": "12"}, "query_timestep"),
        ({"query_timestep": True}, "query_timestep"),
        ({"evidence_timestamps": [8, 4]}, "evidence_timestamps"),
        ({"evidence_timestamps": []}, "evidence_timestamps"),
        ({"deterministic_checks": {"identifiers_valid": "yes"}}, "fields differ"),
        ({"source_request_sha256": "0" * 63}, "source_request_sha256"),
        ({"source_request_sha256": "A" * 64}, "source_request_sha256"),
        ({"suite_name": "imitation"}, "suite_name does not match"),
        ({"episode_id": "BinFill/episode_x"}, "episode_id"),
        ({"candidate_id": "candidate-0"}, "candidate_id"),
        ({"question": "   "}, "question"),
        ({"judge_result": {"evidence_timestamps": [4]}}, "fields differ"),
        ({"judge_result": None}, "judge result must be an object"),
        ({"rejection_reason": "made_up"}, "rejection_reason must be null"),
        ({"schema_version": "halo.robomme.vqa.record/v2"}, "schema_version"),
    ],
)
def test_record_validation_rejects_wrong_value_types(
    mutation: Mapping[str, Any], match: str
) -> None:
    record = {**_valid_record(), **mutation}
    with pytest.raises(ContractError, match=match):
        parse_record(record)


def test_record_validation_rejects_truthy_non_boolean_deterministic_checks() -> None:
    record = _valid_record()
    record["deterministic_checks"] = {**record["deterministic_checks"], "count_valid": 1}
    with pytest.raises(ContractError, match="count_valid must be boolean"):
        parse_record(record)


def _audit_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_robomme_vqa_cli", REPOSITORY_ROOT / "scripts/vqa/audit_robomme_vqa.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_cli_reads_records_from_bytes_and_rejects_truthy_strings() -> None:
    cli = _audit_cli()
    valid = _valid_record()
    payload = json.dumps(valid, sort_keys=True).encode("utf-8")
    assert cli.read_records(payload) == [valid]

    rejected = json.dumps({**valid, "accepted": "false"}, sort_keys=True)
    with pytest.raises(ValueError, match="line 2: accepted must be boolean"):
        cli.read_records(payload + b"\n" + rejected.encode("utf-8"))
    with pytest.raises(ValueError, match="records JSONL is empty"):
        cli.read_records(b"")
