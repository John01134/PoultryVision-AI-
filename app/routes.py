from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import json
import uuid

from .config import settings
from ml.pipeline import VideoPipeline

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "OK"}


@router.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: int = Form(settings.FPS_SAMPLE),
    conf_thresh: float = Form(settings.CONF_THRESH),
    iou_thresh: float = Form(settings.IOU_THRESH),
):
    """
    Upload a video, run bird detection + tracking,
    and return analytics + annotated video URL.
    """

    # ------------------------------------------------------------------
    # Resolve project directories
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUTS_DIR = BASE_DIR / "outputs"

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Save uploaded video safely
    # ------------------------------------------------------------------
    file_ext = Path(file.filename).suffix or ".mp4"
    safe_name = f"input_{uuid.uuid4().hex}{file_ext}"
    video_path = DATA_DIR / safe_name

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    if not video_path.exists():
        raise HTTPException(status_code=500, detail="Uploaded video was not saved")

    # ------------------------------------------------------------------
    # Output video (STATIC SERVED)
    # ------------------------------------------------------------------
    output_filename = "annotated_video.mp4"
    output_video_path = OUTPUTS_DIR / output_filename

    # ------------------------------------------------------------------
    # Run ML pipeline
    # ------------------------------------------------------------------
    pipeline = VideoPipeline(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        fps_sample=fps_sample,
    )

    try:
        result = pipeline.process_video(
            video_path=str(video_path),
            output_path=str(output_video_path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    # ------------------------------------------------------------------
    # CRITICAL FIX: Return URL, NOT file path
    # ------------------------------------------------------------------
    result["artifacts"]["annotated_video"] = f"outputs/{output_filename}"

    # Optional: save response JSON
    with open(OUTPUTS_DIR / "sample_response.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    return JSONResponse(content=result)
