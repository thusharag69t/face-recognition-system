"""
Root entrypoint for face similarity matching.
Forwards call to src.matcher.
"""
from src.matcher import cosine_similarity, euclidean_distance, match_face

__all__ = ["cosine_similarity", "euclidean_distance", "match_face"]
