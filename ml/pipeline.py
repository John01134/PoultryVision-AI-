import cv2
import os
import subprocess
from typing import Dict

from ml.detector import YOLODetector
from ml.tracker import SimpleMOTTracker
from ml.weight_estimator import estimate_weights
from utils.draw_utils import draw_annotations
from utils.video_utils import get_video_meta


class VideoPipeline:
    def __init__(self, conf_thresh=0.25, iou_thresh=0.3, fps_sample=5):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.fps_sample = fps_sample

        self.detector = YOLODetector(conf_thresh=conf_thresh)
        self.tracker = SimpleMOTTracker(
            iou_thresh=iou_thresh,
            max_age_frames=int(2 * fps_sample)
        )

    def process_video(self, video_path: str, output_path: str) -> Dict:
        meta = get_video_meta(video_path)
        cap = cv2.VideoCapture(video_path)

        width = int(meta["width"])
        height = int(meta["height"])
        orig_fps = meta["fps"]
        image_area = width * height

        frame_step = max(1, round(orig_fps / self.fps_sample))

        # ------------------------------------------------------------------
        # 1️ Write TEMP video (OpenCV)
        # ------------------------------------------------------------------
        temp_path = output_path.replace(".mp4", "_raw.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            temp_path,
            fourcc,
            self.fps_sample,
            (width, height)
        )

        counts = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame_idx)

                active_ids = [t.id for t in tracks]
                timestamp_sec = int(frame_idx / orig_fps)

                counts.append({
                    "timestamp_sec": timestamp_sec,
                    "count": len(active_ids)
                })

                annotated = draw_annotations(
                    frame.copy(),
                    tracks,
                    count=len(active_ids)
                )

                writer.write(annotated)

            frame_idx += 1

        cap.release()
        writer.release()

        # ------------------------------------------------------------------
        # 2️ Convert to BROWSER-SAFE MP4 (H.264)
        # ------------------------------------------------------------------
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.remove(temp_path)

        # ------------------------------------------------------------------
        # Weight estimation
        # ------------------------------------------------------------------
        tracks_list = list(self.tracker.tracks.values())
        weight_info = estimate_weights(tracks_list, image_area)

        tracks_sample = []
        for tr in tracks_list[:10]:
            tracks_sample.append({
                "track_id": tr.id,
                "bbox": [float(v) for v in tr.bbox],
                "confidence": float(tr.conf)
            })

        return {
            "video_name": os.path.basename(video_path),
            "fps_sampled": self.fps_sample,
            "counts_over_time": counts,
            "tracks_sample": tracks_sample,
            "weight_estimates": weight_info,
            "artifacts": {
                "annotated_video": "outputs/annotated_video.mp4"
            }
        }
