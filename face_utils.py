"""
Root entrypoint for face utility functions.
Forwards call to src.face_utils.
"""
from src.face_utils import get_face_encodings, load_image, load_image_from_bytes

__all__ = ["load_image", "load_image_from_bytes", "get_face_encodings"]
