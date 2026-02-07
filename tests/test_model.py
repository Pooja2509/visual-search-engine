"""
Tests for api/model.py — verify the EmbeddingExtractor works correctly.
"""
import numpy as np
import pytest
from PIL import Image

from api.model import EmbeddingExtractor
from api.config import EMBEDDING_DIM


# -------------------------------------------------------
# Fixture: Create extractor ONCE, reuse across all tests
# -------------------------------------------------------
@pytest.fixture(scope="module")
def extractor():
    """Load the model once for all tests in this module."""
    ext = EmbeddingExtractor()
    ext.load()
    return ext


# -------------------------------------------------------
# Fixture: Create a fake test image (no need for real data)
# -------------------------------------------------------
@pytest.fixture
def dummy_image():
    """Create a random 100x100 RGB image for testing."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestEmbeddingExtractor:
    """Test the EmbeddingExtractor class."""

    def test_load_sets_flag(self, extractor):
        """After load(), is_loaded should be True."""
        assert extractor.is_loaded is True

    def test_model_is_set(self, extractor):
        """After load(), model should not be None."""
        assert extractor.model is not None

    def test_transform_is_set(self, extractor):
        """After load(), transform should not be None."""
        assert extractor.transform is not None

    def test_device_is_set(self, extractor):
        """After load(), device should be set."""
        assert extractor.device is not None

    def test_extract_returns_numpy(self, extractor, dummy_image):
        """extract() should return a numpy array."""
        embedding = extractor.extract(dummy_image)
        assert isinstance(embedding, np.ndarray), "Should return numpy array"

    def test_extract_shape(self, extractor, dummy_image):
        """Embedding should be shape (1, 2048)."""
        embedding = extractor.extract(dummy_image)
        assert embedding.shape == (1, EMBEDDING_DIM), \
            f"Expected (1, {EMBEDDING_DIM}), got {embedding.shape}"

    def test_extract_is_normalized(self, extractor, dummy_image):
        """Embedding should be L2 normalized (magnitude ≈ 1.0)."""
        embedding = extractor.extract(dummy_image)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, \
            f"Embedding norm should be ~1.0, got {norm}"

    def test_extract_deterministic(self, extractor, dummy_image):
        """Same image should produce same embedding."""
        emb1 = extractor.extract(dummy_image)
        emb2 = extractor.extract(dummy_image)
        assert np.allclose(emb1, emb2, atol=1e-6), \
            "Same image should produce identical embeddings"

    def test_different_images_different_embeddings(self, extractor):
        """Different images should produce different embeddings."""
        img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))     # all black
        img2 = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255) # all white
        emb1 = extractor.extract(img1)
        emb2 = extractor.extract(img2)
        assert not np.allclose(emb1, emb2, atol=1e-3), \
            "Black and white images should have different embeddings"

    def test_grayscale_image_works(self, extractor):
        """Grayscale images should be auto-converted to RGB."""
        gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
        embedding = extractor.extract(gray)
        assert embedding.shape == (1, EMBEDDING_DIM), \
            "Grayscale image should still produce correct shape"

    def test_small_image_works(self, extractor):
        """Very small images should still work (transform resizes them)."""
        tiny = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        embedding = extractor.extract(tiny)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_large_image_works(self, extractor):
        """Large images should work (transform resizes them down)."""
        big = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))
        embedding = extractor.extract(big)
        assert embedding.shape == (1, EMBEDDING_DIM)
