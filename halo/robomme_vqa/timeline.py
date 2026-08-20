"""Deterministic, causal evidence-frame selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TimelineFrame:
    timestep: int
    front_image: str
    wrist_image: str
    event_boundary: bool = False
    change_score: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestep, bool)
            or not isinstance(self.timestep, int)
            or self.timestep < 0
        ):
            raise ValueError("timestep must be a non-negative integer")
        if (
            not isinstance(self.front_image, str)
            or not self.front_image
            or not isinstance(self.wrist_image, str)
            or not self.wrist_image
        ):
            raise ValueError("both front and wrist image references are required")
        if (
            isinstance(self.change_score, bool)
            or not isinstance(self.change_score, (int, float))
            or not math.isfinite(self.change_score)
            or self.change_score < 0
        ):
            raise ValueError("change_score must be non-negative")


def _evenly_spaced(frames: Sequence[TimelineFrame], count: int) -> list[TimelineFrame]:
    if count <= 0 or not frames:
        return []
    if count >= len(frames):
        return list(frames)
    if count == 1:
        return [frames[len(frames) // 2]]
    last = len(frames) - 1
    indices = [round(position * last / (count - 1)) for position in range(count)]
    return [frames[index] for index in indices]


def select_causal_evidence(
    timeline: Sequence[TimelineFrame],
    *,
    query_timestep: int,
    max_frames: int,
) -> tuple[TimelineFrame, ...]:
    """Select event-salient and temporally distributed frames at or before query.

    Event annotations and frame-change scores influence selection only. They are
    intentionally absent from the returned visual prompt representation.
    """
    if (
        isinstance(query_timestep, bool)
        or not isinstance(query_timestep, int)
        or query_timestep < 0
    ):
        raise ValueError("query_timestep must be a non-negative integer")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    timestamps = [frame.timestep for frame in timeline]
    if timestamps != sorted(set(timestamps)):
        raise ValueError("timeline timesteps must be strictly increasing and unique")

    eligible = [frame for frame in timeline if frame.timestep <= query_timestep]
    if not eligible:
        raise ValueError("timeline contains no frame at or before query_timestep")
    if len(eligible) <= max_frames:
        return tuple(eligible)

    selected: dict[int, TimelineFrame] = {eligible[-1].timestep: eligible[-1]}
    event_frames = sorted(
        (frame for frame in eligible[:-1] if frame.event_boundary),
        key=lambda frame: (-frame.change_score, -frame.timestep),
    )
    for frame in event_frames:
        if len(selected) == max_frames:
            break
        selected[frame.timestep] = frame

    remaining = [frame for frame in eligible[:-1] if frame.timestep not in selected]
    for frame in _evenly_spaced(remaining, max_frames - len(selected)):
        selected[frame.timestep] = frame

    if len(selected) < max_frames:
        for frame in sorted(remaining, key=lambda item: (-item.change_score, -item.timestep)):
            selected[frame.timestep] = frame
            if len(selected) == max_frames:
                break
    return tuple(selected[timestep] for timestep in sorted(selected))
