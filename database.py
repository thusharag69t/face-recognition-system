"""
Root entrypoint for database management.
Forwards call to src.database.
"""
from src.database import FaceDatabase

__all__ = ["FaceDatabase"]
