"""Fail-closed joins between VQA records and their source visual requests."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import strict_json_loads
from .io import REQUEST_SCHEMA_VERSION, parse_request_jsonl
from .pipeline import VQARequest


REQUEST_BUILD_MANIFEST_SCHEMA_VERSION = "halo.robomme.vqa.request-build-manifest/v1"


@dataclass(frozen=True)
class ManifestImage:
    encoded_sha256: str
    task_name: str
    episode_id: str
    timestep: int
    camera: str


@dataclass(frozen=True)
class RequestArtifact:
    requests: Mapping[str, VQARequest]
    images: Mapping[Path, ManifestImage]
    requests_sha256: str
    requests_size_bytes: int
    manifest_sha256: str
    manifest_size_bytes: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def parse_request_index(text: str) -> dict[str, VQARequest]:
    """Index strict v1 requests by the hash embedded in generated records."""
    index: dict[str, VQARequest] = {}
    for request_number, request in enumerate(parse_request_jsonl(text), start=1):
        source_hash = request.source_sha256()
        if source_hash in index:
            raise ValueError(
                f"duplicate source_request_sha256 at request {request_number}: "
                f"{source_hash}"
            )
        index[source_hash] = request
    if not index:
        raise ValueError("requests JSONL is empty")
    return index


def read_request_index(path: str | Path) -> dict[str, VQARequest]:
    return parse_request_index(Path(path).read_text(encoding="utf-8"))


def read_request_artifact(
    requests_path: str | Path, manifest_path: str | Path
) -> RequestArtifact:
    """Validate the companion build manifest and index its requests and images."""
    requests_path = Path(requests_path).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    # Every source file is read exactly once; the bytes that are hashed are the
    # bytes that are parsed, so no later re-read can substitute other content.
    manifest_bytes = manifest_path.read_bytes()
    requests_bytes = requests_path.read_bytes()
    payload = strict_json_loads(manifest_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request build manifest must be an object")
    if payload.get("schema_version") != REQUEST_BUILD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported request build manifest schema_version")

    declared_requests = payload.get("requests_jsonl")
    if not isinstance(declared_requests, dict):
        raise ValueError("request build manifest has no requests_jsonl object")
    declared_path = declared_requests.get("path")
    if not isinstance(declared_path, str) or not Path(declared_path).is_absolute():
        raise ValueError("request build manifest requests_jsonl path must be absolute")
    if Path(declared_path).resolve(strict=False) != requests_path:
        raise ValueError("request build manifest requests_jsonl path mismatch")
    declared_hash = _sha256_value(
        declared_requests.get("sha256"), "request build manifest requests_jsonl sha256"
    )
    requests_sha256 = _sha256_bytes(requests_bytes)
    if requests_sha256 != declared_hash:
        raise ValueError("requests JSONL SHA-256 does not match request build manifest")
    size_bytes = declared_requests.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes != len(requests_bytes)
    ):
        raise ValueError("requests JSONL size does not match request build manifest")
    if declared_requests.get("request_schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("request build manifest has unsupported request schema_version")

    requests = parse_request_index(requests_bytes.decode("utf-8"))
    counts = payload.get("counts")
    if not isinstance(counts, dict) or counts.get("requests") != len(requests):
        raise ValueError("request count does not match request build manifest")

    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("request build manifest images must be a non-empty list")
    if counts.get("images") != len(raw_images):
        raise ValueError("image count does not match request build manifest")

    artifact_root = manifest_path.parent.resolve()
    images: dict[Path, ManifestImage] = {}
    for image_number, raw_image in enumerate(raw_images, start=1):
        if not isinstance(raw_image, dict):
            raise ValueError(f"request build manifest image {image_number} is not an object")
        relative_path = raw_image.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(
                f"request build manifest image {image_number} has invalid relative_path"
            )
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(
                f"request build manifest image {image_number} path must be relative"
            )
        absolute = (artifact_root / relative).resolve(strict=False)
        if artifact_root not in absolute.parents:
            raise ValueError(
                f"request build manifest image {image_number} escapes artifact root"
            )
        if absolute in images:
            raise ValueError(f"duplicate request build manifest image path: {relative_path}")

        task_name = raw_image.get("task_name")
        episode_id = raw_image.get("episode_id")
        timestep = raw_image.get("timestep")
        camera = raw_image.get("camera")
        if not isinstance(task_name, str) or not isinstance(episode_id, str):
            raise ValueError(
                f"request build manifest image {image_number} has invalid identity"
            )
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError(
                f"request build manifest image {image_number} has invalid timestep"
            )
        if camera not in {"front", "wrist"}:
            raise ValueError(
                f"request build manifest image {image_number} has invalid camera"
            )
        images[absolute] = ManifestImage(
            encoded_sha256=_sha256_value(
                raw_image.get("encoded_sha256"),
                f"request build manifest image {image_number} encoded_sha256",
            ),
            task_name=task_name,
            episode_id=episode_id,
            timestep=timestep,
            camera=camera,
        )

    _verify_timeline_images(requests, images)
    return RequestArtifact(
        requests=requests,
        images=images,
        requests_sha256=requests_sha256,
        requests_size_bytes=len(requests_bytes),
        manifest_sha256=_sha256_bytes(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
    )


def _verify_timeline_images(
    requests: Mapping[str, VQARequest], images: Mapping[Path, ManifestImage]
) -> None:
    """Bind every image referenced by every request, not only audited evidence."""
    for source_hash, request in requests.items():
        for frame in request.timeline:
            for camera, reference in (
                ("front", frame.front_image),
                ("wrist", frame.wrist_image),
            ):
                label = (
                    f"request {source_hash} {camera} image at timestep {frame.timestep}"
                )
                image_path = Path(reference)
                if not image_path.is_absolute():
                    raise ValueError(f"{label} must be an absolute path")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                resolved = image_path.resolve(strict=True)
                provenance = images.get(resolved)
                if provenance is None:
                    raise ValueError(f"{label} is absent from request build manifest")
                if (
                    provenance.task_name,
                    provenance.episode_id,
                    provenance.timestep,
                    provenance.camera,
                ) != (request.task_name, request.episode_id, frame.timestep, camera):
                    raise ValueError(f"{label} metadata does not match its source request")
                if _sha256_bytes(resolved.read_bytes()) != provenance.encoded_sha256:
                    raise ValueError(
                        f"{label} SHA-256 does not match request build manifest"
                    )


def _evidence_timestamps(record: Mapping[str, Any]) -> tuple[set[int], set[int], int]:
    candidate = record.get("evidence_timestamps")
    judge_result = record.get("judge_result")
    judge = (
        judge_result.get("evidence_timestamps")
        if isinstance(judge_result, dict)
        else None
    )
    query = record.get("query_timestep")
    for label, value in (
        ("candidate evidence_timestamps", candidate),
        ("judge evidence_timestamps", judge),
    ):
        if (
            not isinstance(value, list)
            or not value
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value
            )
            or value != sorted(set(value))
        ):
            raise ValueError(f"record has invalid {label}")
    if isinstance(query, bool) or not isinstance(query, int) or query < 0:
        raise ValueError("record has invalid query_timestep")
    return set(candidate), set(judge), query


def attach_visual_evidence(
    record: Mapping[str, Any], artifact: RequestArtifact
) -> dict[str, Any]:
    """Copy a record and attach its required, read-only image references."""
    source_hash = record.get("source_request_sha256")
    if not isinstance(source_hash, str) or source_hash not in artifact.requests:
        raise ValueError(f"record has no matching source request: {source_hash!r}")
    request = artifact.requests[source_hash]
    if request.source_sha256() != source_hash:
        raise ValueError("request index key does not match request content hash")

    expected = {
        "task_name": request.task_name,
        "suite_name": request.suite_name,
        "episode_id": request.episode_id,
        "query_timestep": request.query_timestep,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"record/request {field} mismatch for {source_hash}: "
                f"{record.get(field)!r} != {value!r}"
            )

    candidate, judge, query = _evidence_timestamps(record)
    required = candidate | judge | {query}
    timeline = {frame.timestep: frame for frame in request.timeline}
    missing = sorted(required - set(timeline))
    if missing:
        raise ValueError(
            f"record evidence is unavailable in source request {source_hash}: {missing}"
        )

    # Image references need no re-check here: read_request_artifact already bound
    # every timeline image in this request to the build manifest and its bytes.
    visual_evidence: list[dict[str, Any]] = []
    for timestep in sorted(required):
        frame = timeline[timestep]
        visual_evidence.append(
            {
                "timestep": timestep,
                "front_image": frame.front_image,
                "wrist_image": frame.wrist_image,
                "candidate_evidence": timestep in candidate,
                "judge_evidence": timestep in judge,
                "query_frame": timestep == query,
            }
        )
    return {**record, "visual_evidence": visual_evidence}


__all__ = [
    "ManifestImage",
    "RequestArtifact",
    "attach_visual_evidence",
    "parse_request_index",
    "read_request_artifact",
    "read_request_index",
]
