"""Deterministic filtering independent of VLM judgment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from halo.robomme.dataset import TASK_TO_SUITE

from .contracts import Candidate, JudgeResult
from .prompts import allowed_families


PRIVILEGED_TERM_PATTERN = re.compile(
    r"\b(?:is_video_demo|is_subgoal_boundary|simple_subgoal|grounded_subgoal|"
    r"joint_action|gripper_action|joint_state|gripper_state|simulator(?:[_ -]?id)?|"
    r"object[_ -]?id|body[_ -]?id|geom[_ -]?id|privileged[_ -]?label|hdf5)\b",
    flags=re.IGNORECASE,
)
EPISODE_PATTERN = re.compile(r"^(?P<task>[A-Za-z0-9]+)/episode_(?P<number>[0-9]+)$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TIMESTAMP_REFERENCE_PATTERN = re.compile(
    r"\b(?:frame|timestep)\s*(?:[:=#-]\s*)?([0-9]+)\b", flags=re.IGNORECASE
)
COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass(frozen=True)
class ValidationContext:
    task_name: str
    suite_name: str
    episode_id: str
    query_timestep: int
    permitted_timestamps: tuple[int, ...]


@dataclass(frozen=True)
class DeterministicChecks:
    identifiers_valid: bool
    timestamps_valid: bool
    no_privileged_terms: bool
    count_valid: bool
    not_duplicate: bool
    answer_within_bounds: bool

    @property
    def passes(self) -> bool:
        return all(self.to_dict().values())

    def to_dict(self) -> dict[str, bool]:
        return {
            "identifiers_valid": self.identifiers_valid,
            "timestamps_valid": self.timestamps_valid,
            "no_privileged_terms": self.no_privileged_terms,
            "count_valid": self.count_valid,
            "not_duplicate": self.not_duplicate,
            "answer_within_bounds": self.answer_within_bounds,
        }


class EpisodeDeduplicator:
    """Deterministic exact and near-duplicate rejection within each episode."""

    def __init__(self, *, jaccard_threshold: float = 0.9) -> None:
        if not 0 < jaccard_threshold <= 1:
            raise ValueError("jaccard_threshold must be in (0, 1]")
        self.jaccard_threshold = jaccard_threshold
        self._questions: dict[str, list[tuple[str, frozenset[str]]]] = {}

    @staticmethod
    def _normalized(question: str) -> tuple[str, frozenset[str]]:
        tokens = TOKEN_PATTERN.findall(question.lower())
        return " ".join(tokens), frozenset(tokens)

    def is_duplicate(self, episode_id: str, question: str) -> bool:
        normalized, tokens = self._normalized(question)
        for previous, previous_tokens in self._questions.get(episode_id, []):
            if normalized == previous:
                return True
            union = tokens | previous_tokens
            similarity = len(tokens & previous_tokens) / len(union) if union else 1.0
            if similarity >= self.jaccard_threshold:
                return True
        return False

    def remember(self, episode_id: str, question: str) -> None:
        self._questions.setdefault(episode_id, []).append(self._normalized(question))


def _identifiers_valid(context: ValidationContext, candidate: Candidate) -> bool:
    match = EPISODE_PATTERN.fullmatch(context.episode_id)
    return (
        TASK_TO_SUITE.get(context.task_name) == context.suite_name
        and match is not None
        and match.group("task") == context.task_name
        and candidate.question_family in allowed_families(context.task_name)
    )


def _timestamps_valid(
    context: ValidationContext,
    candidate: Candidate,
    judge: JudgeResult,
) -> bool:
    permitted = set(context.permitted_timestamps)
    timestamp_sets = (candidate.evidence_timestamps, judge.evidence_timestamps)
    referenced = [
        int(value)
        for value in TIMESTAMP_REFERENCE_PATTERN.findall(
            f"{candidate.question}\n{candidate.answer}"
        )
    ]
    return (
        context.permitted_timestamps == tuple(sorted(set(context.permitted_timestamps)))
        and bool(permitted)
        and max(permitted) <= context.query_timestep
        and all(set(timestamps).issubset(permitted) for timestamps in timestamp_sets)
        and all(max(timestamps) <= context.query_timestep for timestamps in timestamp_sets)
        and all(timestamp <= context.query_timestep for timestamp in referenced)
        and any(timestamp < context.query_timestep for timestamp in candidate.evidence_timestamps)
        and any(timestamp < context.query_timestep for timestamp in judge.evidence_timestamps)
    )


def _extract_single_count(answer: str) -> int | None:
    tokens = TOKEN_PATTERN.findall(answer.lower())
    values: list[int] = []
    for token in tokens:
        if token.isdigit():
            values.append(int(token))
        elif token in COUNT_WORDS:
            values.append(COUNT_WORDS[token])
    return values[0] if len(values) == 1 else None


def _answer_within_bounds(answer: str, *, maximum_words: int = 24) -> bool:
    return (
        0 < len(answer) <= 128
        and len(answer.split()) <= maximum_words
        and answer.isprintable()
        and "\n" not in answer
    )


def validate_candidate(
    candidate: Candidate,
    judge: JudgeResult,
    context: ValidationContext,
    *,
    deduplicator: EpisodeDeduplicator,
    known_count: int | None = None,
) -> DeterministicChecks:
    """Run checks and remember only candidates that pass every non-dedupe check."""
    identifiers_valid = _identifiers_valid(context, candidate)
    timestamps_valid = _timestamps_valid(context, candidate, judge)
    no_privileged_terms = PRIVILEGED_TERM_PATTERN.search(
        f"{candidate.question}\n{candidate.answer}"
    ) is None
    count_valid = known_count is None or _extract_single_count(candidate.answer) == known_count
    not_duplicate = not deduplicator.is_duplicate(context.episode_id, candidate.question)
    answer_within_bounds = _answer_within_bounds(candidate.answer)
    checks = DeterministicChecks(
        identifiers_valid=identifiers_valid,
        timestamps_valid=timestamps_valid,
        no_privileged_terms=no_privileged_terms,
        count_valid=count_valid,
        not_duplicate=not_duplicate,
        answer_within_bounds=answer_within_bounds,
    )
    if checks.passes:
        deduplicator.remember(context.episode_id, candidate.question)
    return checks


def rejection_reason(checks: DeterministicChecks, judge: JudgeResult) -> str | None:
    for name, passed in checks.to_dict().items():
        if not passed:
            return f"deterministic_{name}"
    if not judge.passes:
        return f"judge_{judge.reason_code}"
    return None
