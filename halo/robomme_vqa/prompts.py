"""RoboMME task and memory-question prompts for offline VQA generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from halo.robomme.dataset import TASK_TO_SUITE

from .contracts import CANDIDATE_SCHEMA_VERSION, JUDGE_SCHEMA_VERSION, Candidate
from .timeline import TimelineFrame


@dataclass(frozen=True)
class QuestionFamily:
    name: str
    instruction: str


QUESTION_FAMILIES: dict[str, QuestionFamily] = {
    "event_count": QuestionFamily(
        "event_count", "Ask how many task-relevant events have completed by the query time."
    ),
    "event_ordinality": QuestionFamily(
        "event_ordinality", "Ask which occurrence or cycle has most recently completed."
    ),
    "motion_order": QuestionFamily(
        "motion_order", "Ask about the earlier ordering or direction of a repeated motion."
    ),
    "last_seen_location": QuestionFamily(
        "last_seen_location", "Ask where a task-relevant object was last visible before occlusion."
    ),
    "identity_tracking": QuestionFamily(
        "identity_tracking",
        "Ask which currently visible container or object corresponds to an earlier identity.",
    ),
    "highlight_identity": QuestionFamily(
        "highlight_identity", "Ask which object was highlighted during the earlier brief cue."
    ),
    "action_referent": QuestionFamily(
        "action_referent", "Ask which object was manipulated earlier and must be selected again."
    ),
    "temporal_referent": QuestionFamily(
        "temporal_referent", "Ask which target is identified by an earlier button-relative event."
    ),
    "ordinal_referent": QuestionFamily(
        "ordinal_referent", "Ask which target occurred at a specified earlier ordinal position."
    ),
    "contact_mode": QuestionFamily(
        "contact_mode", "Ask which demonstrated manipulation strategy or tool contact was used."
    ),
    "peg_orientation": QuestionFamily(
        "peg_orientation", "Ask which peg end, grasp end, or insertion direction was demonstrated."
    ),
    "trajectory_order": QuestionFamily(
        "trajectory_order", "Ask for the ordered targets in the demonstrated path."
    ),
    "route_direction": QuestionFamily(
        "route_direction", "Ask for the demonstrated route order or circling direction."
    ),
}


SUITE_INSTRUCTIONS: dict[str, str] = {
    "counting": (
        "Focus on temporal counts, completed repetitions, motion order, and the occurrence "
        "at which an action must stop."
    ),
    "permanence": (
        "Focus on object identity and location before occlusion and on tracking containers "
        "through visible swaps."
    ),
    "reference": (
        "Focus on persistent identity under visual, action-based, temporal, or ordinal cues."
    ),
    "imitation": (
        "Focus on the demonstrated manipulation strategy, contact mode, orientation, and "
        "ordered motion path."
    ),
}


TASK_INSTRUCTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "BinFill": (
        "Track how many requested-color cubes have been placed into the opaque bin.",
        ("event_count", "event_ordinality"),
    ),
    "PickXtimes": (
        "Track completed pick-and-place repetitions for the instructed cube.",
        ("event_count", "event_ordinality"),
    ),
    "SwingXtimes": (
        "Track completed swing cycles and the prior target-to-target direction.",
        ("event_count", "event_ordinality", "motion_order"),
    ),
    "StopCube": (
        "Track the moving cube's target crossings so the specified occurrence is identifiable.",
        ("event_count", "event_ordinality", "motion_order"),
    ),
    "VideoUnmask": (
        "Remember the requested cube's location before all cubes are covered.",
        ("last_seen_location", "identity_tracking"),
    ),
    "ButtonUnmask": (
        "Remember cube locations before the button-triggered masking event.",
        ("last_seen_location", "identity_tracking"),
    ),
    "VideoUnmaskSwap": (
        "Track the requested cube's container while the visible containers swap.",
        ("last_seen_location", "identity_tracking", "motion_order"),
    ),
    "ButtonUnmaskSwap": (
        "Track the requested cube's container during button-triggered masking and swaps.",
        ("last_seen_location", "identity_tracking", "motion_order"),
    ),
    "PickHighlight": (
        "Remember which cubes received the earlier brief visual highlight.",
        ("highlight_identity", "identity_tracking"),
    ),
    "VideoRepick": (
        "Remember the cube manipulated in the demonstration and its subsequent identity.",
        ("action_referent", "identity_tracking", "event_count"),
    ),
    "VideoPlaceButton": (
        "Resolve the target identified relative to an earlier button press.",
        ("temporal_referent", "identity_tracking"),
    ),
    "VideoPlaceOrder": (
        "Resolve the target identified by its earlier placement order.",
        ("ordinal_referent", "identity_tracking"),
    ),
    "MoveCube": (
        "Remember whether the demonstration used grasping, pushing, or a stick hook.",
        ("contact_mode", "action_referent"),
    ),
    "InsertPeg": (
        "Remember the demonstrated peg, grasp end, and insertion direction.",
        ("peg_orientation", "action_referent"),
    ),
    "PatternLock": (
        "Remember the full ordered linear target trajectory.",
        ("trajectory_order", "motion_order"),
    ),
    "RouteStick": (
        "Remember the ordered targets and circling directions around the sticks.",
        ("route_direction", "trajectory_order"),
    ),
}

if set(TASK_INSTRUCTIONS) != set(TASK_TO_SUITE):
    raise RuntimeError("VQA task prompts must cover exactly the 16 RoboMME tasks")


def allowed_families(task_name: str) -> tuple[str, ...]:
    try:
        return TASK_INSTRUCTIONS[task_name][1]
    except KeyError as error:
        raise ValueError(f"unknown RoboMME task: {task_name}") from error


def _base_context(
    *, task_name: str, suite_name: str, task_goal: str, query_timestep: int
) -> str:
    if TASK_TO_SUITE.get(task_name) != suite_name:
        raise ValueError("suite_name does not match task_name")
    task_instruction, family_names = TASK_INSTRUCTIONS[task_name]
    family_lines = "\n".join(
        f"- {name}: {QUESTION_FAMILIES[name].instruction}" for name in family_names
    )
    return f"""Task: {task_name}
Suite: {suite_name}
Task goal: {task_goal}
Query timestep: {query_timestep}
Suite memory focus: {SUITE_INSTRUCTIONS[suite_name]}
Task-specific focus: {task_instruction}
Allowed question families:
{family_lines}"""


def generator_prompt(
    *,
    task_name: str,
    suite_name: str,
    task_goal: str,
    query_timestep: int,
    evidence: Sequence[TimelineFrame],
    candidate_count: int,
) -> str:
    if not 1 <= candidate_count <= 8:
        raise ValueError("candidate_count must be between 1 and 8")
    timestamps = [frame.timestep for frame in evidence]
    if not timestamps or max(timestamps) > query_timestep:
        raise ValueError("generator evidence must be non-empty and causal")
    if not any(timestep < query_timestep for timestep in timestamps):
        raise ValueError("generator evidence must include history before the query")
    context = _base_context(
        task_name=task_name,
        suite_name=suite_name,
        task_goal=task_goal,
        query_timestep=query_timestep,
    )
    return f"""You construct concise visual-memory VQA pairs for robot-policy supervision.
Use only the timestamped front and wrist images supplied in this call. Never infer from
future frames, simulator state, metadata field names, privileged labels, or hidden chain
of thought. Each question must require at least one frame strictly earlier than timestep
{query_timestep}; a question answerable from only the current image is invalid.

{context}

Return exactly {candidate_count} distinct candidates as one JSON object and no prose:
{{
  "schema_version": "{CANDIDATE_SCHEMA_VERSION}",
  "candidates": [
    {{
      "candidate_id": "candidate-1",
      "question_family": "one allowed family",
      "question": "memory-dependent question",
      "answer": "concise answer of at most 24 words",
      "evidence_timestamps": [{timestamps[0]}]
    }}
  ]
}}
Candidate identifiers must be consecutive. Evidence timestamps must be a sorted unique
subset of {timestamps}, must not exceed {query_timestep}, and must include an earlier
frame actually needed to answer the question. Mention camera identity when it matters."""


def judge_prompt(
    *,
    task_name: str,
    suite_name: str,
    task_goal: str,
    query_timestep: int,
    evidence: Sequence[TimelineFrame],
    candidate: Candidate,
) -> str:
    timestamps = [frame.timestep for frame in evidence]
    if not timestamps or max(timestamps) > query_timestep:
        raise ValueError("judge evidence must be non-empty and causal")
    if not any(timestep < query_timestep for timestep in timestamps):
        raise ValueError("judge evidence must include history before the query")
    context = _base_context(
        task_name=task_name,
        suite_name=suite_name,
        task_goal=task_goal,
        query_timestep=query_timestep,
    )
    candidate_payload = {
        "candidate_id": candidate.candidate_id,
        "question_family": candidate.question_family,
        "question": candidate.question,
        "answer": candidate.answer,
        "claimed_evidence_timestamps": list(candidate.evidence_timestamps),
    }
    return f"""You are an independent deterministic visual-grounding judge. Evaluate the
candidate from scratch using only the images and task context in this call. You receive
no generator reasoning and must not invent it. Mark history_required true only if a
strictly earlier frame is necessary. Mark future_leakage true if the candidate depends
on anything after query timestep {query_timestep} or on unavailable privileged metadata.

{context}

Candidate:
{json.dumps(candidate_payload, sort_keys=True)}

Return one JSON object and no prose:
{{
  "schema_version": "{JUDGE_SCHEMA_VERSION}",
  "visually_answerable": true,
  "history_required": true,
  "answer_correct": true,
  "unambiguous": true,
  "future_leakage": false,
  "evidence_timestamps": [{timestamps[0]}],
  "reason_code": "accepted"
}}
Evidence timestamps must be a sorted unique subset of {timestamps}. Use a short lower-case
snake_case reason code; do not output reasoning or any extra field."""


def multimodal_messages(prompt: str, evidence: Sequence[TimelineFrame]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
    for frame in evidence:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"timestep={frame.timestep}, camera=front",
                },
                {"type": "image", "image": frame.front_image},
                {
                    "type": "text",
                    "text": f"timestep={frame.timestep}, camera=wrist",
                },
                {"type": "image", "image": frame.wrist_image},
            ]
        )
    return [{"role": "user", "content": content}]
