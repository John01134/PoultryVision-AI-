from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routes import router

# --------------------------------------------------
# Resolve project root correctly
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Bird Counting & Weight Estimation",
    description="Detection, tracking, counting and weight proxy estimation for poultry videos",
    version="1.0.0",
)

# --------------------------------------------------
# API routes
# --------------------------------------------------
app.include_router(router)

# --------------------------------------------------
# Serve annotated videos (STATIC)
# URL: http://127.0.0.1:8000/outputs/annotated_video.mp4
# --------------------------------------------------
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)

# --------------------------------------------------
# Serve frontend UI
# URL: http://127.0.0.1:8000/
# --------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    return FileResponse(index_file)
