import cv2
from typing import Tuple, Dict, List


def get_video_meta(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": fps, "width": width, "height": height, "frame_count": frame_count}


def write_video(path: str, frames: List, fps: int = 5):
    if len(frames) == 0:
        # create an empty file to indicate no output
        open(path, "wb").close()
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

