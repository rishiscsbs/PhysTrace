"""SRS section 5 data schemas (Pydantic v2)."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["SENSOR", "TWIN_PREDICTION", "WATERMARK", "CAMERA_ESTIMATE", "CONTROLLER_LOG", "NETWORK_MONITOR"]
Quality = Literal["GOOD", "UNCERTAIN", "LOW_QUALITY", "UNAVAILABLE", "BAD"]
# Provenance classes shown on every UI badge / chart trace / record (FR-033).
Provenance = Literal["SENSOR_OBSERVATION", "CAMERA_VISUAL_ESTIMATE", "DIGITAL_TWIN_PREDICTION", "INFERENCE", "SIMULATION_GROUND_TRUTH"]

HypothesisType = Literal[
    "SENSOR_SPOOFING",
    "SENSOR_FAILURE",
    "COMMUNICATION_MANIPULATION_OR_DELAY",
    "CONTROLLER_MANIPULATION",
    "ACTUATOR_FAULT",
    "LEGITIMATE_DISTURBANCE",
    "TWIN_MODEL_MISMATCH",
    "INCONCLUSIVE",
]
HYPOTHESIS_TYPES: tuple[str, ...] = HypothesisType.__args__  # type: ignore[attr-defined]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Observation(BaseModel):
    """SRS 5.1"""

    observation_id: str = Field(default_factory=lambda: new_id("OBS"))
    timestamp: float
    source_type: SourceType
    source_id: str
    component_id: str
    measurement_type: str
    value: float | str | bool | None
    unit: str
    quality: Quality = "GOOD"
    uncertainty: float | None = None
    freshness: float = 0.0  # seconds since the underlying measurement was produced
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """SRS 5.2"""

    event_id: str = Field(default_factory=lambda: new_id("EVT"))
    run_id: str
    timestamp: float
    severity: Literal["INFO", "NOTICE", "WARNING", "CRITICAL"] = "INFO"
    category: str
    component_id: str | None = None
    message: str
    provenance: Provenance = "INFERENCE"
    data: dict[str, Any] = Field(default_factory=dict)


class Experiment(BaseModel):
    """SRS 5.3"""

    experiment_id: str = Field(default_factory=lambda: new_id("EXP"))
    kind: Literal["COUNTERFACTUAL", "ABLATION", "LIVE_RUN"]
    created_at: float
    status: Literal["PENDING", "RUNNING", "COMPLETE", "FAILED"] = "PENDING"
    config: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class Hypothesis(BaseModel):
    """SRS 5.4. Scores are uncalibrated EVIDENCE SCORES, never probabilities."""

    hypothesis_id: str = Field(default_factory=lambda: new_id("HYP"))
    hypothesis_type: HypothesisType
    component_id: str | None = None
    score: float  # normalised evidence share in [0, 1] (not a probability)
    raw_score: float
    score_type: Literal["EVIDENCE_SCORE"] = "EVIDENCE_SCORE"
    rank: int = 0
    status: Literal["LEADING", "COMPETING", "DISFAVOURED", "INCONCLUSIVE", "NOT_EVALUATED"] = "COMPETING"
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    contradicting_evidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    explanation: str = ""


class CameraCalibration(BaseModel):
    """SRS 5.5"""

    calibration_id: str = Field(default_factory=lambda: new_id("CAL"))
    camera_id: str = "CAM-01"
    component_id: str = "TK-101"
    mode: str
    roi: dict[str, int]  # x, y, w, h in pixels
    points: list[dict[str, float]]  # two points: {"pixel_y": .., "height_m": ..}
    indicator_roi: dict[str, int] | None = None
    created_at: float = 0.0
    notes: str = ""


class EvidenceItem(BaseModel):
    """A normalised, scorable piece of evidence derived from one or more Observations."""

    evidence_id: str
    feature: str
    component_id: str
    strength: float  # 0..1
    value: float | str | None = None
    description: str
    independence_group: str
    observation_refs: list[str] = Field(default_factory=list)
    source_type: SourceType
    provenance: Provenance = "INFERENCE"
    timestamp: float
