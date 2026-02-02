"""
Search module — loads FAISS index and finds similar images.
"""
import numpy as np
import pandas as pd
import faiss
from api.config import INDEX_PATH, IDS_PATH, METADATA_PATH, EMBEDDING_DIM


class SearchEngine:
    """
    Handles loading the FAISS index and searching for similar images.

    Usage:
        engine = SearchEngine()
        engine.load()
        results = engine.search(query_embedding, top_k=5)
    """

    def __init__(self):
        self.index = None       # FAISS index (will be loaded)
        self.ids = None         # image IDs array
        self.metadata = None    # product metadata DataFrame
        self.is_loaded = False  # flag: has everything been loaded?

    def load(self):
        """Load FAISS index, IDs, and metadata from disk."""
        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        print(f"  Index loaded: {self.index.ntotal:,} vectors")

        print("Loading image IDs...")
        self.ids = np.load(IDS_PATH)
        print(f"  IDs loaded: {len(self.ids):,}")

        print("Loading metadata...")
        self.metadata = pd.read_csv(METADATA_PATH)
        # Create a fast lookup dictionary: id → row of metadata
        self.id_to_meta = {}
        for _, row in self.metadata.iterrows():
            self.id_to_meta[row['id']] = {
                'id': int(row['id']),
                'articleType': row['articleType'],
                'masterCategory': row['masterCategory'],
                'subCategory': row['subCategory'],
                'productDisplayName': row.get('productDisplayName', 'Unknown'),
                'baseColour': row.get('baseColour', 'Unknown'),
            }
        print(f"  Metadata loaded: {len(self.id_to_meta):,} products")

        self.is_loaded = True
        print("Search engine ready!")

    def search(self, query_embedding, top_k=10):
        """
        Search for similar images.

        Args:
            query_embedding: numpy array of shape (1, 2048) — L2 normalized
            top_k: number of results to return

        Returns:
            list of dictionaries with product info and similarity score
        """
        if not self.is_loaded:
            raise RuntimeError("Search engine not loaded. Call .load() first.")

        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k + 1)

        # Build results list
        results = []
        for rank in range(top_k + 1):
            idx = indices[0][rank]
            image_id = int(self.ids[idx])
            distance = float(distances[0][rank])

            # Skip the query itself (distance = 0)
            if distance < 1e-6:
                continue

            # Convert L2 distance to cosine similarity
            cosine_sim = 1.0 - distance / 2.0

            # Get metadata
            meta = self.id_to_meta.get(image_id, {})

            results.append({
                'id': image_id,
                'score': round(cosine_sim, 4),
                'articleType': meta.get('articleType', 'Unknown'),
                'masterCategory': meta.get('masterCategory', 'Unknown'),
                'productDisplayName': meta.get('productDisplayName', 'Unknown'),
                'baseColour': meta.get('baseColour', 'Unknown'),
                'image_url': f"/images/{image_id}.jpg",
            })

            if len(results) >= top_k:
                break

        return results
