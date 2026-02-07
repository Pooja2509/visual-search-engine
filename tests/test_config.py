"""
Tests for api/config.py — verify all configuration paths and settings.
"""
import os
import pytest
from api.config import (
    BASE_DIR, EMBEDDINGS_PATH, IDS_PATH, METADATA_PATH,
    INDEX_PATH, MODEL_PATH, IMAGE_DIR,
    TOP_K, EMBEDDING_DIM, IMAGE_SIZE, CROP_SIZE,
    IMAGENET_MEAN, IMAGENET_STD
)


class TestPaths:
    """Test that all configured paths are valid."""

    def test_base_dir_exists(self):
        """BASE_DIR should point to the project root."""
        assert os.path.isdir(BASE_DIR), f"BASE_DIR does not exist: {BASE_DIR}"

    def test_base_dir_contains_api(self):
        """BASE_DIR should contain the api/ folder."""
        api_dir = os.path.join(BASE_DIR, "api")
        assert os.path.isdir(api_dir), f"api/ folder not found in {BASE_DIR}"

    def test_embeddings_path_format(self):
        """EMBEDDINGS_PATH should end with .npy."""
        assert EMBEDDINGS_PATH.endswith(".npy"), "Embeddings should be a .npy file"

    def test_ids_path_format(self):
        """IDS_PATH should end with .npy."""
        assert IDS_PATH.endswith(".npy"), "IDs should be a .npy file"

    def test_metadata_path_format(self):
        """METADATA_PATH should end with .csv."""
        assert METADATA_PATH.endswith(".csv"), "Metadata should be a .csv file"

    def test_index_path_format(self):
        """INDEX_PATH should end with .faiss."""
        assert INDEX_PATH.endswith(".faiss"), "Index should be a .faiss file"

    def test_model_path_format(self):
        """MODEL_PATH should end with .pth."""
        assert MODEL_PATH.endswith(".pth"), "Model should be a .pth file"


class TestSettings:
    """Test that all settings have valid values."""

    def test_top_k_is_positive(self):
        """TOP_K must be a positive integer."""
        assert isinstance(TOP_K, int), "TOP_K should be an integer"
        assert TOP_K > 0, "TOP_K should be positive"

    def test_embedding_dim(self):
        """EMBEDDING_DIM should be 2048 (ResNet50 output)."""
        assert EMBEDDING_DIM == 2048, "ResNet50 produces 2048-dim embeddings"

    def test_image_size(self):
        """IMAGE_SIZE should be 256 (standard resize)."""
        assert IMAGE_SIZE == 256

    def test_crop_size(self):
        """CROP_SIZE should be 224 (standard for ImageNet models)."""
        assert CROP_SIZE == 224

    def test_crop_smaller_than_resize(self):
        """CROP_SIZE must be <= IMAGE_SIZE (crop after resize)."""
        assert CROP_SIZE <= IMAGE_SIZE, "Can't crop larger than the resized image"

    def test_imagenet_mean_length(self):
        """ImageNet mean should have 3 values (R, G, B)."""
        assert len(IMAGENET_MEAN) == 3, "Need 3 channel means"

    def test_imagenet_std_length(self):
        """ImageNet std should have 3 values (R, G, B)."""
        assert len(IMAGENET_STD) == 3, "Need 3 channel stds"

    def test_imagenet_mean_range(self):
        """ImageNet mean values should be between 0 and 1."""
        for val in IMAGENET_MEAN:
            assert 0.0 < val < 1.0, f"Mean {val} should be between 0 and 1"

    def test_imagenet_std_range(self):
        """ImageNet std values should be between 0 and 1."""
        for val in IMAGENET_STD:
            assert 0.0 < val < 1.0, f"Std {val} should be between 0 and 1"
