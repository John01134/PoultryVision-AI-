import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    a1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    a2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class Track:
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], conf: float, frame_idx: int):
        self.id = track_id
        self.bbox = bbox
        self.conf = conf
        self.last_frame = frame_idx
        self.hits = 1
        self.history = [bbox]


class SimpleMOTTracker:
    """A straightforward IOU-based MOT tracker using Hungarian assignment.

    Not a production ByteTrack implementation, but provides stable IDs and
    handles occlusions with simple aging.
    """

    def __init__(self, iou_thresh: float = 0.3, max_age_frames: int = 50):
        self.iou_thresh = iou_thresh
        self.max_age_frames = max_age_frames
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float, float]], frame_idx: int):
        """detections: list of (x1,y1,x2,y2,conf)"""
        det_boxes = [d[:4] for d in detections]
        det_confs = [d[4] for d in detections]

        if len(self.tracks) == 0:
            for b, c in zip(det_boxes, det_confs):
                t = Track(self._next_id, b, c, frame_idx)
                self.tracks[self._next_id] = t
                self._next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t].bbox for t in track_ids]

        if len(det_boxes) == 0:
            # nothing detected: age tracks and remove old ones
            self._age_tracks(frame_idx)
            return list(self.tracks.values())

        cost_matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                cost_matrix[i, j] = 1.0 - iou(tb, db)

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        # Apply matches
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= 1.0 - self.iou_thresh:
                tid = track_ids[r]
                self.tracks[tid].bbox = det_boxes[c]
                self.tracks[tid].conf = det_confs[c]
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].hits += 1
                self.tracks[tid].history.append(det_boxes[c])
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # Create new tracks for unassigned detections
        for j, (b, c) in enumerate(zip(det_boxes, det_confs)):
            if j not in assigned_dets:
                t = Track(self._next_id, b, c, frame_idx)
                self.tracks[self._next_id] = t
                self._next_id += 1

        # Age and remove stale tracks
        self._age_tracks(frame_idx)

        return list(self.tracks.values())

    def _age_tracks(self, frame_idx: int):
        to_remove = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_frame > self.max_age_frames:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

