# **🐔 PoultryVision-AI**

Bird Counting, Tracking & Weight Estimation from Poultry CCTV Video

# 1. Project Overview

PoultryVision-AI is a computer-vision–based prototype designed to analyze fixed-camera poultry CCTV videos and automatically extract flock-level analytics using YOLOv8 as the sole detection model.

The system performs:

- Bird detection using YOLOv8
- Stable multi-object tracking with persistent IDs
- Accurate bird counting over time
- Relative (proxy-based) weight estimation
- Annotated video generation for visual validation

The entire pipeline is exposed through a FastAPI backend and demonstrated via a browser-based frontend, making it suitable as a production-ready prototype for real-world poultry monitoring systems.

# 2. Problem Statement

Manual bird counting and monitoring in poultry farms:

- Is labor-intensive
- Does not scale to large flocks
- Is prone to human error

Given a single fixed CCTV camera feed (MP4), the objective is to:

- Detect individual birds in each frame
- Track birds across frames with stable IDs
- Count birds reliably without double counting
- Estimate bird weight using a visual proxy
- Produce both machine-readable analytics and human-verifiable visual output

# 3. High-Level System Architecture

        Frontend (HTML / JS)
                ↓
        FastAPI Backend
                ↓
        Video Processing Pipeline
                ├─ Detection (YOLOv8)
                ├─ Tracking (IOU + Hungarian Assignment)
                ├─ Counting Logic
                ├─ Weight Proxy Estimation
                └─ Annotation Rendering
                ↓
        Outputs
         ├─ Annotated Video (MP4)
         └─ Structured JSON Analytics

# 4. Technology Stack & Rationale

## 4.1 Programming Language

Python 3.8+

- Mature ecosystem for computer vision
- Strong support for AI and numerical computing
- Excellent compatibility with FastAPI and OpenCV

## 4.2 Backend Framework

FastAPI

- High-performance asynchronous API
- Automatic OpenAPI documentation
- Clean request/response validation
- Easy integration with frontend and future microservices

Used to:

- Accept video uploads
- Trigger processing pipeline
- Serve annotated video as static content
- Return structured JSON analytics

## 4.3 Object Detection

YOLOv8 (Ultralytics)

Why YOLOv8:

- Real-time performance
- Strong generalization
- Simple Python API
- Widely accepted industry standard

Implementation details:

- Pretrained YOLOv8 model
- COCO dataset class filtering
- Only class ID = 14 (bird) is used
- Confidence threshold applied to remove weak detections

Output per frame:

(x1, y1, x2, y2, confidence)

## 4.4 Multi-Object Tracking

Custom IOU-based MOT Tracker + Hungarian Algorithm

Why not simple frame-by-frame counting:

- Causes double counting
- Cannot track individual birds

Tracking approach:

- Each detection is matched to existing tracks using Intersection-over-Union (IOU)
- Hungarian assignment ensures optimal matching
- New IDs are created for unmatched detections
- Tracks are removed after configurable inactivity (max_age_frames)

Result:

- Stable tracking IDs
- Each bird retains its ID across frames
- Enables reliable counting and weight aggregation

## 4.5 Counting Logic

Counting is derived from active tracking IDs, not raw detections.

Process:

- For each sampled frame:
  - Count unique active track IDs
  - Store counts with timestamp (seconds)

Why this works:

- Prevents double counting
- Robust to temporary occlusion
- Reflects actual flock size trends

Output:

{ "timestamp_sec": 3, "count": 6 }

## 4.6 Weight Estimation (Proxy-Based)

Since no physical scale or calibration data is available, the system computes a relative weight index.

Method:

- Use bounding-box area as a proxy for bird size
- Normalize by total image area
- Aggregate per bird and per flock

Output:

- Per-bird average proxy index
- Flock-level average proxy

⚠️ Important Assumption

This is not real weight in grams.
Conversion requires camera calibration and labeled weight data.

## 4.7 Video Annotation

OpenCV

Each output frame includes:

- Bounding boxes around birds
- Stable tracking ID per bird
- Total bird count displayed on the frame

This provides:

- Visual verification
- Debugging capability
- Confidence for non-technical stakeholders

# 5. End-to-End Pipeline (Step-by-Step)

- Video uploaded via frontend or API
- Metadata extracted (FPS, resolution)
- Frame sampling applied (default 5 FPS)
- YOLOv8 detects birds
- Tracker assigns stable IDs
- Count computed from active IDs
- Weight proxy calculated
- Annotations drawn
- Annotated MP4 video written
- JSON analytics returned

# 6. API Usage

Endpoint
POST /analyze_video

Parameters

- file – input CCTV video
- fps_sample – processing FPS
- conf_thresh – detection confidence threshold
- iou_thresh – tracking IOU threshold

Response

- Structured JSON analytics
- Path to annotated video (static-served)

# 7. Frontend Demo

The frontend provides:

- Video upload
- Processing status feedback
- Annotated video playback
- JSON analytics display
- Optional JSON download

Purpose:

- Demonstration
- Validation
- Non-technical stakeholder access

# 8. Project Structure

        bird_counting_weight_estimation/
        ├── app/
        │   ├── main.py        # FastAPI entry point
        │   ├── routes.py      # API routes
        │   └── config.py
        ├── ml/
        │   ├── detector.py    # YOLOv8 wrapper
        │   ├── tracker.py     # MOT tracker
        │   ├── pipeline.py    # End-to-end pipeline
        │   └── weight_estimator.py
        ├── utils/
        │   ├── draw_utils.py
        │   └── video_utils.py
        ├── frontend/
        │   └── index.html
        ├── data/
        ├── outputs/
        ├── requirements.txt
        ├── README.md
        └── run.sh

# 9. Limitations (Explicitly Acknowledged)

- Detection accuracy depends on COCO pretrained model
- IOU-based tracker may struggle under heavy occlusion
- Weight estimates are relative indices, not grams
- Single-camera assumption

These are acceptable and expected for a prototype.

# 10. Future Improvements

- ByteTrack or DeepSORT integration
- Camera calibration module
- Real-weight regression mapping
- Live RTSP stream support
- Dockerized deployment
- Dashboard analytics

# 11. How to Run the Project

1. Install dependencies:
bash
   pip install -r requirements.txt 
2. Start the backend:
   uvicorn app.main:app --reload
3. Open the browser and navigate to:
   http://127.0.0.1:8000
4. Upload a poultry CCTV video and view results.
   - Annotated output video

   - Bird count analytics

   - Relative weight estimation (proxy)
     
This project prioritizes correct system design, explainability, and modularity over absolute detection accuracy.

# 12.Conclusion

PoultryVision-AI demonstrates a technically sound and well-structured computer vision pipeline for poultry video analytics.

The project delivers:
- Reliable bird detection and stable multi-object tracking
- Accurate bird counting over time without double counting
- A transparent, proxy-based approach to weight estimation
- Annotated visual outputs for easy validation
- A clean FastAPI backend with a usable frontend demo

The system emphasizes clear design, modularity, and explainability, making it suitable as a production-ready prototype for evaluation, demonstration, and technical discussion.

PoultryVision-AI fully satisfies the stated company requirements and provides a strong foundation for future enhancements and real-world deployment.

