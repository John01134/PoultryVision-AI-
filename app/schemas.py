from pydantic import BaseModel
from typing import List, Dict, Any


class CountPoint(BaseModel):
    timestamp_sec: int
    count: int


class TrackSample(BaseModel):
    track_id: int
    bbox: List[float]
    confidence: float


class WeightEstimates(BaseModel):
    unit: str
    per_bird_avg: Dict[str, float]
    flock_avg: float
    assumptions: str


class Artifacts(BaseModel):
    annotated_video: str


class AnalyzeResponse(BaseModel):
    video_name: str
    fps_sampled: int
    counts_over_time: List[CountPoint]
    tracks_sample: List[TrackSample]
    weight_estimates: WeightEstimates
    artifacts: Artifacts

