"""
FastAPI REST API Backend Server for Face Recognition Identification System.

Provides endpoints for:
  - System status & database inspection
  - Enrolling new face images
  - Query face identification with annotated image overlays
  - Identity management & enrolled photos inspection / lightbox
  - Threshold evaluation sweep metrics and curves
  - Config management
  - Web UI static asset hosting
"""
import os
import sys
import io
import base64
import glob
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.database import FaceDatabase
from src.face_utils import load_image_from_bytes, load_image
from src.enroll import enroll_single_image
from src.identify import identify_image, draw_annotations, encode_image_base64
from src.evaluate import run_evaluation

app = FastAPI(
    title="Face Recognition System API",
    description="REST API for enrolling, identifying, and evaluating face recognition metrics.",
    version="1.1.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database instance
db = FaceDatabase(config.DATABASE_PATH)


class ConfigUpdateModel(BaseModel):
    metric: Optional[str] = None
    threshold: Optional[float] = None


@app.get("/api/status")
def get_system_status():
    people = db.list_people()
    details = db.get_identity_details()
    total_embeddings = sum(details.values())
    return {
        "status": "online",
        "database_path": config.DATABASE_PATH,
        "enrolled_identities": len(people),
        "total_embeddings": total_embeddings,
        "identities": details,
        "detection_model": config.DETECTION_MODEL,
        "metric": config.SIMILARITY_METRIC,
        "threshold": config.MATCH_THRESHOLD,
    }


@app.post("/api/enroll")
async def enroll_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")

    contents = await file.read()
    image_rgb = load_image_from_bytes(contents)
    if image_rgb is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

    success = enroll_single_image(name, image_rgb, db, save_file=True)
    if not success:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No usable face detected for '{name}'. Keep one face centered, "
                "move closer, improve lighting, and capture again."
            ),
        )

    details = db.get_identity_details()
    return {
        "success": True,
        "message": f"Successfully enrolled face for '{name}'.",
        "person_name": name,
        "total_photos_for_person": details.get(name, 1),
        "total_identities": len(db.list_people()),
    }


@app.post("/api/identify")
async def identify_face(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    metric: Optional[str] = Form(None)
):
    contents = await file.read()
    image_rgb = load_image_from_bytes(contents)
    if image_rgb is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

    use_metric = metric if metric in ("cosine", "euclidean") else config.SIMILARITY_METRIC
    use_threshold = threshold if threshold is not None else (
        config.COSINE_THRESHOLD if use_metric == "cosine" else config.EUCLIDEAN_THRESHOLD
    )

    results = identify_image(
        image_rgb,
        db=db,
        threshold=use_threshold,
        metric=use_metric
    )

    annotated_rgb = draw_annotations(image_rgb, results)
    b64_image = encode_image_base64(annotated_rgb)

    return {
        "face_count": len(results),
        "threshold_used": use_threshold,
        "metric_used": use_metric,
        "faces": results,
        "annotated_image_base64": b64_image,
    }


@app.get("/api/identities")
def list_identities():
    people = db.list_people()
    details = db.get_identity_details()

    identities_list = []
    for name in people:
        person_dir = os.path.join(config.ENROLLED_IMAGES_DIR, name)
        photo_count = details.get(name, 0)
        first_photo_b64 = ""

        if os.path.exists(person_dir):
            files = [
                f for f in sorted(os.listdir(person_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if files:
                first_path = os.path.join(person_dir, files[0])
                first_img = load_image(first_path)
                if first_img is not None:
                    first_photo_b64 = encode_image_base64(first_img)

        identities_list.append({
            "name": name,
            "embedding_count": photo_count,
            "thumbnail_base64": first_photo_b64,
        })

    return {"identities": identities_list}


@app.get("/api/identities/{name}/photos")
def get_identity_photos(name: str):
    person_dir = os.path.join(config.ENROLLED_IMAGES_DIR, name)
    if not os.path.exists(person_dir):
        return {"name": name, "photos": []}

    photos = []
    files = [
        f for f in sorted(os.listdir(person_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    for fname in files:
        fpath = os.path.join(person_dir, fname)
        img_rgb = load_image(fpath)
        if img_rgb is not None:
            b64 = encode_image_base64(img_rgb)
            stat = os.stat(fpath)
            photos.append({
                "filename": fname,
                "size_kb": round(stat.st_size / 1024, 1),
                "url": f"/api/identities/{name}/photos/{fname}",
                "base64": b64,
            })

    return {"name": name, "count": len(photos), "photos": photos}


@app.get("/api/identities/{name}/photos/{filename}")
def serve_identity_photo(name: str, filename: str):
    fpath = os.path.join(config.ENROLLED_IMAGES_DIR, name, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Photo file not found.")
    return FileResponse(fpath)


@app.delete("/api/identities/{name}")
def delete_identity(name: str):
    removed = db.remove_person(name)
    person_dir = os.path.join(config.ENROLLED_IMAGES_DIR, name)
    if os.path.exists(person_dir):
        import shutil
        shutil.rmtree(person_dir, ignore_errors=True)

    if not removed and not os.path.exists(person_dir):
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    return {"success": True, "message": f"Deleted identity '{name}' and all associated photos."}


@app.delete("/api/identities/{name}/photos/{filename}")
def delete_identity_photo(name: str, filename: str):
    fpath = os.path.join(config.ENROLLED_IMAGES_DIR, name, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
        return {"success": True, "message": f"Deleted photo '{filename}' for identity '{name}'."}
    raise HTTPException(status_code=404, detail="Photo file not found.")


@app.get("/api/evaluate")
def run_or_get_evaluation(metric: Optional[str] = Query("cosine")):
    try:
        eval_res = run_evaluation(metric=metric)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    plot_path = os.path.join(config.RESULTS_DIR, "threshold_curve.png")

    plot_b64 = ""
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as f:
            plot_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "summary": eval_res,
        "plot_base64": plot_b64,
    }


@app.post("/api/config")
def update_config(data: ConfigUpdateModel):
    if data.metric in ("cosine", "euclidean"):
        config.SIMILARITY_METRIC = data.metric
    if data.threshold is not None:
        config.MATCH_THRESHOLD = float(data.threshold)
        if config.SIMILARITY_METRIC == "cosine":
            config.COSINE_THRESHOLD = float(data.threshold)
        else:
            config.EUCLIDEAN_THRESHOLD = float(data.threshold)

    return {
        "success": True,
        "metric": config.SIMILARITY_METRIC,
        "threshold": config.MATCH_THRESHOLD,
    }


# Serve static web frontend
os.makedirs(config.STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Face Recognition API Server on http://localhost:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
