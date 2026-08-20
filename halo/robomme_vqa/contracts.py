"""Strict v1 JSON contracts and pinned model configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Sequence


CANDIDATE_SCHEMA_VERSION = "halo.robomme.vqa.candidates/v1"
JUDGE_SCHEMA_VERSION = "halo.robomme.vqa.judge/v1"
RECORD_SCHEMA_VERSION = "halo.robomme.vqa.record/v1"
MODEL_CONFIG_SCHEMA_VERSION = "halo.robomme.vqa.model-config/v1"
APPROVED_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
APPROVED_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
APPROVED_RUNTIME_VERSIONS = {
    "torch": "2.9.1+cu128",
    "transformers": "5.15.1",
    "qwen-vl-utils": "0.0.14",
}


class ContractError(ValueError):
    """Raised when model output does not satisfy the strict v1 contract."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def strict_json_loads(text: str) -> Any:
    """Parse one JSON value while rejecting duplicate keys and NaN/Infinity."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except JSONDecodeError as error:
        raise ContractError(f"invalid JSON: {error}") from error


def extract_json_object(text: str, *, required_key: str) -> dict[str, Any]:
    """Extract one matching JSON object from noisy VLM output.

    Markdown fences and surrounding prose are tolerated. Duplicate keys,
    non-finite values, arrays at the top level, and ambiguous multiple matching
    objects fail closed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ContractError("model output is empty")
    if len(text) > 1_000_000:
        raise ContractError("model output exceeds 1 MB")

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    matches: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except (JSONDecodeError, ContractError):
            continue
        if isinstance(value, dict) and required_key in value:
            matches.append(value)
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one JSON object containing {required_key!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty_string(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ContractError(f"{label} must be a non-empty string of at most {maximum} chars")
    return value.strip()


def _timestamps(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ContractError(f"{label} must be a non-empty list of non-negative integers")
    if value != sorted(set(value)):
        raise ContractError(f"{label} must be strictly increasing and unique")
    return tuple(value)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    question_family: str
    question: str
    answer: str
    evidence_timestamps: tuple[int, ...]


@dataclass(frozen=True)
class JudgeResult:
    visually_answerable: bool
    history_required: bool
    answer_correct: bool
    unambiguous: bool
    future_leakage: bool
    evidence_timestamps: tuple[int, ...]
    reason_code: str

    @property
    def passes(self) -> bool:
        return (
            self.visually_answerable
            and self.history_required
            and self.answer_correct
            and self.unambiguous
            and not self.future_leakage
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JUDGE_SCHEMA_VERSION,
            "visually_answerable": self.visually_answerable,
            "history_required": self.history_required,
            "answer_correct": self.answer_correct,
            "unambiguous": self.unambiguous,
            "future_leakage": self.future_leakage,
            "evidence_timestamps": list(self.evidence_timestamps),
            "reason_code": self.reason_code,
        }


def parse_candidates(text: str) -> tuple[Candidate, ...]:
    payload = extract_json_object(text, required_key="candidates")
    _exact_keys(payload, {"schema_version", "candidates"}, "candidate response")
    if payload["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ContractError("unsupported candidate schema_version")
    rows = payload["candidates"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ContractError("candidates must contain between 1 and 8 items")

    parsed: list[Candidate] = []
    identifiers: set[str] = set()
    expected = {
        "candidate_id",
        "question_family",
        "question",
        "answer",
        "evidence_timestamps",
    }
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ContractError("each candidate must be an object")
        _exact_keys(row, expected, f"candidate {index}")
        candidate_id = _nonempty_string(row["candidate_id"], "candidate_id", maximum=32)
        if candidate_id != f"candidate-{index}":
            raise ContractError(
                "candidate_id values must be consecutive candidate-1, candidate-2, ..."
            )
        if candidate_id in identifiers:
            raise ContractError(f"duplicate candidate_id: {candidate_id}")
        identifiers.add(candidate_id)
        parsed.append(
            Candidate(
                candidate_id=candidate_id,
                question_family=_nonempty_string(
                    row["question_family"], "question_family", maximum=64
                ),
                question=_nonempty_string(row["question"], "question", maximum=512),
                answer=_nonempty_string(row["answer"], "answer", maximum=128),
                evidence_timestamps=_timestamps(
                    row["evidence_timestamps"], "candidate evidence_timestamps"
                ),
            )
        )
    return tuple(parsed)


def parse_judge_result(text: str) -> JudgeResult:
    payload = extract_json_object(text, required_key="visually_answerable")
    expected = {
        "schema_version",
        "visually_answerable",
        "history_required",
        "answer_correct",
        "unambiguous",
        "future_leakage",
        "evidence_timestamps",
        "reason_code",
    }
    _exact_keys(payload, expected, "judge response")
    if payload["schema_version"] != JUDGE_SCHEMA_VERSION:
        raise ContractError("unsupported judge schema_version")
    for field in (
        "visually_answerable",
        "history_required",
        "answer_correct",
        "unambiguous",
        "future_leakage",
    ):
        if not isinstance(payload[field], bool):
            raise ContractError(f"{field} must be boolean")
    reason_code = _nonempty_string(payload["reason_code"], "reason_code", maximum=64)
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason_code) is None:
        raise ContractError("reason_code must be lower-case snake_case")
    return JudgeResult(
        visually_answerable=payload["visually_answerable"],
        history_required=payload["history_required"],
        answer_correct=payload["answer_correct"],
        unambiguous=payload["unambiguous"],
        future_leakage=payload["future_leakage"],
        evidence_timestamps=_timestamps(
            payload["evidence_timestamps"], "judge evidence_timestamps"
        ),
        reason_code=reason_code,
    )


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_revision: str
    processor_revision: str
    dtype: str
    attention_implementation: str
    torch_version: str
    transformers_version: str
    qwen_vl_utils_version: str
    image_min_pixels: int
    image_max_pixels: int
    video_min_pixels: int
    video_max_pixels: int
    max_images_per_call: int
    max_video_frames: int
    generator_prompt_revision: str
    judge_prompt_revision: str
    generator_generation: dict[str, Any]
    judge_generation: dict[str, Any]

    def model_provenance(self, *, role: str) -> dict[str, Any]:
        if role not in {"generator", "judge"}:
            raise ValueError("role must be generator or judge")
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "dtype": self.dtype,
            "attention_implementation": self.attention_implementation,
            "runtime_versions": {
                "torch": self.torch_version,
                "transformers": self.transformers_version,
                "qwen-vl-utils": self.qwen_vl_utils_version,
            },
            "visual_limits": {
                "image_min_pixels": self.image_min_pixels,
                "image_max_pixels": self.image_max_pixels,
                "video_min_pixels": self.video_min_pixels,
                "video_max_pixels": self.video_max_pixels,
                "max_images_per_call": self.max_images_per_call,
                "max_video_frames": self.max_video_frames,
            },
            "prompt_revision": getattr(self, f"{role}_prompt_revision"),
            "generation": dict(getattr(self, f"{role}_generation")),
        }

    def verify_runtime(self) -> dict[str, str]:
        expected = {
            "torch": self.torch_version,
            "transformers": self.transformers_version,
            "qwen-vl-utils": self.qwen_vl_utils_version,
        }
        observed: dict[str, str] = {}
        for package, expected_version in expected.items():
            try:
                observed[package] = version(package)
            except PackageNotFoundError as error:
                raise RuntimeError(f"required VQA package is not installed: {package}") from error
            if observed[package] != expected_version:
                raise RuntimeError(
                    f"{package} runtime mismatch: expected {expected_version}, "
                    f"found {observed[package]}"
                )
        return observed


def load_model_config(path: str | Path) -> ModelConfig:
    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "model_id",
        "model_revision",
        "processor_revision",
        "dtype",
        "attention_implementation",
        "torch_version",
        "transformers_version",
        "qwen_vl_utils_version",
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "max_images_per_call",
        "max_video_frames",
        "generator_prompt_revision",
        "judge_prompt_revision",
        "generator_generation",
        "judge_generation",
    }
    if not isinstance(payload, dict):
        raise ContractError("model config must be an object")
    _exact_keys(payload, expected, "model config")
    if payload["schema_version"] != MODEL_CONFIG_SCHEMA_VERSION:
        raise ContractError("unsupported model config schema_version")
    if payload["model_id"] != APPROVED_MODEL_ID:
        raise ContractError(f"model_id must remain pinned to {APPROVED_MODEL_ID}")
    if payload["model_revision"] != APPROVED_REVISION:
        raise ContractError(f"model_revision must remain pinned to {APPROVED_REVISION}")
    if payload["processor_revision"] != APPROVED_REVISION:
        raise ContractError(f"processor_revision must remain pinned to {APPROVED_REVISION}")
    if payload["dtype"] != "float16":
        raise ContractError("the RTX 8000 VQA configuration must use float16")
    if payload["attention_implementation"] != "sdpa":
        raise ContractError("the RTX 8000 VQA configuration must use SDPA attention")
    configured_runtime = {
        "torch": payload["torch_version"],
        "transformers": payload["transformers_version"],
        "qwen-vl-utils": payload["qwen_vl_utils_version"],
    }
    if configured_runtime != APPROVED_RUNTIME_VERSIONS:
        raise ContractError(
            f"VQA runtime must remain pinned to {APPROVED_RUNTIME_VERSIONS}"
        )
    for field in (
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "max_images_per_call",
    ):
        if (
            isinstance(payload[field], bool)
            or not isinstance(payload[field], int)
            or payload[field] <= 0
        ):
            raise ContractError(f"{field} must be a positive integer")
    if payload["image_min_pixels"] > payload["image_max_pixels"]:
        raise ContractError("image pixel limits are reversed")
    if payload["video_min_pixels"] > payload["video_max_pixels"]:
        raise ContractError("video pixel limits are reversed")
    if payload["max_video_frames"] != 0:
        raise ContractError("v1 uses timestamped images and must keep max_video_frames at zero")
    for role in ("generator", "judge"):
        generation = payload[f"{role}_generation"]
        if not isinstance(generation, dict):
            raise ContractError(f"{role}_generation must be an object")
        _exact_keys(
            generation,
            {"do_sample", "max_new_tokens", "temperature"},
            f"{role}_generation",
        )
        if not isinstance(generation["do_sample"], bool):
            raise ContractError(f"{role} do_sample must be boolean")
        if (
            isinstance(generation["max_new_tokens"], bool)
            or not isinstance(generation["max_new_tokens"], int)
            or generation["max_new_tokens"] <= 0
        ):
            raise ContractError(f"{role} max_new_tokens must be a positive integer")
        if (
            isinstance(generation["temperature"], bool)
            or not isinstance(generation["temperature"], (int, float))
            or generation["temperature"] < 0
        ):
            raise ContractError(f"{role} temperature must be a non-negative number")
    if payload["judge_generation"] != {
        "do_sample": False,
        "max_new_tokens": 512,
        "temperature": 0.0,
    }:
        raise ContractError("judge decoding must remain deterministic")
    return ModelConfig(
        model_id=payload["model_id"],
        model_revision=payload["model_revision"],
        processor_revision=payload["processor_revision"],
        dtype=payload["dtype"],
        attention_implementation=payload["attention_implementation"],
        torch_version=payload["torch_version"],
        transformers_version=payload["transformers_version"],
        qwen_vl_utils_version=payload["qwen_vl_utils_version"],
        image_min_pixels=payload["image_min_pixels"],
        image_max_pixels=payload["image_max_pixels"],
        video_min_pixels=payload["video_min_pixels"],
        video_max_pixels=payload["video_max_pixels"],
        max_images_per_call=payload["max_images_per_call"],
        max_video_frames=payload["max_video_frames"],
        generator_prompt_revision=_nonempty_string(
            payload["generator_prompt_revision"], "generator_prompt_revision", maximum=64
        ),
        judge_prompt_revision=_nonempty_string(
            payload["judge_prompt_revision"], "judge_prompt_revision", maximum=64
        ),
        generator_generation=dict(payload["generator_generation"]),
        judge_generation=dict(payload["judge_generation"]),
    )
