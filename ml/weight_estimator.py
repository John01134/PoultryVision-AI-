from typing import Dict, List, Tuple


def compute_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def estimate_weights(tracks: List[object], image_area: float) -> Dict:
    """Compute per-track weight index and flock average.

    Weight index = average(bbox_area / image_area) per track.
    Returns structural dict as required by API.
    """
    per_bird = {}
    vals = []
    for tr in tracks:
        # Use track history to compute mean area ratio
        areas = [compute_area(b) for b in getattr(tr, "history", [tr.bbox])]
        if len(areas) == 0:
            continue
        mean_idx = sum(areas) / len(areas) / image_area
        per_bird[str(tr.id)] = float(mean_idx)
        vals.append(mean_idx)

    flock_avg = float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0

    return {
        "unit": "index",
        "per_bird_avg": per_bird,
        "flock_avg": flock_avg,
        "assumptions": "Bounding box area used as weight proxy; camera calibration required to convert to grams."
    }

