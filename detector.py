"""
Root entrypoint for face detection.
Forwards call to src.detector.
"""
from src.detector import FaceDetector, detect_faces

__all__ = ["FaceDetector", "detect_faces"]
