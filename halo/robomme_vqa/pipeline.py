"""Offline generator -> independent judge -> deterministic-filter pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from halo.robomme.dataset import TASK_TO_SUITE

from .contracts import (
    RECORD_SCHEMA_VERSION,
    Candidate,
    JudgeResult,
    ModelConfig,
    parse_candidates,
    parse_judge_result,
)
from .prompts import allowed_families, generator_prompt, judge_prompt, multimodal_messages
from .qwen import ChatBackend
from .timeline import TimelineFrame, select_causal_evidence
from .validation import (
    EpisodeDeduplicator,
    ValidationContext,
    rejection_reason,
    validate_candidate,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VQARequest:
    task_name: str
    suite_name: str
    episode_id: str
    task_goal: str
    query_timestep: int
    timeline: tuple[TimelineFrame, ...]
    known_counts_by_family: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if TASK_TO_SUITE.get(self.task_name) != self.suite_name:
            raise ValueError("suite_name does not match task_name")
        prefix = f"{self.task_name}/episode_"
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id.startswith(prefix)
            or not self.episode_id[len(prefix) :].isdigit()
        ):
            raise ValueError("episode_id must use TaskName/episode_N")
        if not isinstance(self.task_goal, str) or not self.task_goal.strip():
            raise ValueError("task_goal must be non-empty")
        if (
            isinstance(self.query_timestep, bool)
            or not isinstance(self.query_timestep, int)
            or self.query_timestep < 0
        ):
            raise ValueError("query_timestep must be a non-negative integer")
        if not self.timeline:
            raise ValueError("timeline must be non-empty")
        permitted_families = set(allowed_families(self.task_name))
        for family, count in self.known_counts_by_family.items():
            if family not in permitted_families:
                raise ValueError(f"known count uses disallowed family: {family}")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"known count for {family} must be a non-negative integer")

    def source_sha256(self) -> str:
        payload = {
            "task_name": self.task_name,
            "suite_name": self.suite_name,
            "episode_id": self.episode_id,
            "task_goal": self.task_goal,
            "query_timestep": self.query_timestep,
            "timeline": [
                {
                    "timestep": frame.timestep,
                    "front_image": frame.front_image,
                    "wrist_image": frame.wrist_image,
                    "event_boundary": frame.event_boundary,
                    "change_score": frame.change_score,
                }
                for frame in self.timeline
            ],
            "known_counts_by_family": dict(sorted(self.known_counts_by_family.items())),
        }
        return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class CandidateGenerator:
    """Generator interface with its own prompt and decoding configuration."""

    def __init__(self, backend: ChatBackend, config: ModelConfig) -> None:
        self.backend = backend
        self.config = config

    def call(
        self,
        request: VQARequest,
        evidence: Sequence[TimelineFrame],
        *,
        candidate_count: int,
    ) -> tuple[tuple[Candidate, ...], str]:
        prompt = generator_prompt(
            task_name=request.task_name,
            suite_name=request.suite_name,
            task_goal=request.task_goal,
            query_timestep=request.query_timestep,
            evidence=evidence,
            candidate_count=candidate_count,
        )
        raw = self.backend.generate(
            multimodal_messages(prompt, evidence), self.config.generator_generation
        )
        return parse_candidates(raw), raw


class IndependentJudge:
    """Judge interface; every candidate starts a fresh prompt and model call."""

    def __init__(self, backend: ChatBackend, config: ModelConfig) -> None:
        self.backend = backend
        self.config = config

    def call(
        self,
        request: VQARequest,
        evidence: Sequence[TimelineFrame],
        candidate: Candidate,
    ) -> tuple[JudgeResult, str]:
        prompt = judge_prompt(
            task_name=request.task_name,
            suite_name=request.suite_name,
            task_goal=request.task_goal,
            query_timestep=request.query_timestep,
            evidence=evidence,
            candidate=candidate,
        )
        raw = self.backend.generate(
            multimodal_messages(prompt, evidence), self.config.judge_generation
        )
        return parse_judge_result(raw), raw


class OfflineVQAPipeline:
    """Construct corpus records; this class is never part of policy serving."""

    def __init__(
        self,
        generator: CandidateGenerator,
        judge: IndependentJudge,
        *,
        max_evidence_frames: int = 16,
        deduplicator: EpisodeDeduplicator | None = None,
    ) -> None:
        if max_evidence_frames <= 0:
            raise ValueError("max_evidence_frames must be positive")
        if generator.config != judge.config:
            raise ValueError("generator and judge must use the same pinned model config")
        if max_evidence_frames * 2 > generator.config.max_images_per_call:
            raise ValueError("two-camera evidence exceeds max_images_per_call")
        self.generator = generator
        self.judge = judge
        self.max_evidence_frames = max_evidence_frames
        self.deduplicator = deduplicator or EpisodeDeduplicator()

    def process(self, request: VQARequest, *, candidate_count: int = 3) -> list[dict[str, Any]]:
        evidence = select_causal_evidence(
            request.timeline,
            query_timestep=request.query_timestep,
            max_frames=self.max_evidence_frames,
        )
        candidates, generator_raw = self.generator.call(
            request, evidence, candidate_count=candidate_count
        )
        source_hash = request.source_sha256()
        generator_hash = _sha256_text(generator_raw)
        config = self.generator.config
        context = ValidationContext(
            task_name=request.task_name,
            suite_name=request.suite_name,
            episode_id=request.episode_id,
            query_timestep=request.query_timestep,
            permitted_timestamps=tuple(frame.timestep for frame in evidence),
        )

        records: list[dict[str, Any]] = []
        for candidate in candidates:
            judge_result, judge_raw = self.judge.call(request, evidence, candidate)
            checks = validate_candidate(
                candidate,
                judge_result,
                context,
                deduplicator=self.deduplicator,
                known_count=request.known_counts_by_family.get(candidate.question_family),
            )
            reason = rejection_reason(checks, judge_result)
            records.append(
                {
                    "schema_version": RECORD_SCHEMA_VERSION,
                    "task_name": request.task_name,
                    "suite_name": request.suite_name,
                    "episode_id": request.episode_id,
                    "query_timestep": request.query_timestep,
                    "candidate_id": candidate.candidate_id,
                    "question": candidate.question,
                    "answer": candidate.answer,
                    "question_family": candidate.question_family,
                    "evidence_timestamps": list(candidate.evidence_timestamps),
                    "generator_model_revision": config.model_revision,
                    "generator_prompt_revision": config.generator_prompt_revision,
                    "judge_model_revision": config.model_revision,
                    "judge_prompt_revision": config.judge_prompt_revision,
                    "source_request_sha256": source_hash,
                    "generator_response_sha256": generator_hash,
                    "judge_response_sha256": _sha256_text(judge_raw),
                    "deterministic_checks": checks.to_dict(),
                    "judge_result": judge_result.to_dict(),
                    "accepted": reason is None,
                    "rejection_reason": reason,
                }
            )
        return records
