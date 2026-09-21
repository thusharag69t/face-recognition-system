"""
Root entrypoint for face embedding extraction.
Forwards call to src.embedder.
"""
from src.embedder import FaceEmbedder, extract_embeddings

__all__ = ["FaceEmbedder", "extract_embeddings"]
