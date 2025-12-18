def frame_to_seconds(frame_idx: int, fps: float) -> float:
    return frame_idx / float(fps) if fps > 0 else 0.0

