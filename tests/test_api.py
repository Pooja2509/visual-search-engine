"""
Tests for api/main.py — test the FastAPI endpoints.

Uses FastAPI's TestClient (no real server needed).
"""
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from io import BytesIO

from api.main import app


# -------------------------------------------------------
# Fixture: Create TestClient ONCE, reuse across all tests
# -------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create a test client. This triggers the startup event (loads model)."""
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------
# Fixture: Create a test image as bytes
# -------------------------------------------------------
@pytest.fixture
def test_image_bytes():
    """Create a dummy JPEG image as bytes for upload testing."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        """GET /health should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        """Health response should have status field."""
        data = client.get("/health").json()
        assert "status" in data

    def test_health_is_healthy(self, client):
        """Status should be 'healthy'."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_model_loaded(self, client):
        """Model should be loaded."""
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_index_loaded(self, client):
        """FAISS index should be loaded."""
        data = client.get("/health").json()
        assert data["index_loaded"] is True

    def test_health_index_size(self, client):
        """Index should have 43,916 products."""
        data = client.get("/health").json()
        assert data["index_size"] == 43916


class TestSearchEndpoint:
    """Test the /search endpoint."""

    def test_search_returns_200(self, client, test_image_bytes):
        """POST /search with image should return 200."""
        response = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_search_returns_results(self, client, test_image_bytes):
        """Search should return results list."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_returns_10_results_by_default(self, client, test_image_bytes):
        """Default TOP_K is 10."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        assert data["num_results"] == 10

    def test_search_custom_top_k(self, client, test_image_bytes):
        """top_k parameter should control result count."""
        data = client.post(
            "/search?top_k=5",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        assert data["num_results"] == 5

    def test_search_result_has_required_fields(self, client, test_image_bytes):
        """Each result should have id, score, articleType, image_url."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        result = data["results"][0]
        assert "id" in result
        assert "score" in result
        assert "articleType" in result
        assert "image_url" in result

    def test_search_scores_between_0_and_1(self, client, test_image_bytes):
        """Cosine similarity scores should be between 0 and 1."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        for result in data["results"]:
            assert 0.0 <= result["score"] <= 1.0, \
                f"Score {result['score']} is out of range [0, 1]"

    def test_search_scores_sorted_descending(self, client, test_image_bytes):
        """Results should be sorted by score (highest first)."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True), \
            "Results should be sorted by descending score"

    def test_search_has_timing(self, client, test_image_bytes):
        """Response should include search time."""
        data = client.post(
            "/search",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        ).json()
        assert "search_time_ms" in data
        assert data["search_time_ms"] > 0

    def test_search_rejects_non_image(self, client):
        """Non-image files should be rejected with 400."""
        response = client.post(
            "/search",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        assert response.status_code == 400

    def test_search_no_file_returns_422(self, client):
        """Missing file should return 422 (validation error)."""
        response = client.post("/search")
        assert response.status_code == 422


class TestImageEndpoint:
    """Test the /images/{image_id} endpoint."""

    def test_valid_image_returns_200(self, client):
        """Known image ID should return 200."""
        # First search to get a valid ID
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        data = client.post(
            "/search",
            files={"file": ("test.jpg", buffer.getvalue(), "image/jpeg")}
        ).json()

        image_id = data["results"][0]["id"]
        response = client.get(f"/images/{image_id}")
        assert response.status_code == 200

    def test_valid_image_returns_jpeg(self, client):
        """Image endpoint should return JPEG content."""
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        data = client.post(
            "/search",
            files={"file": ("test.jpg", buffer.getvalue(), "image/jpeg")}
        ).json()

        image_id = data["results"][0]["id"]
        response = client.get(f"/images/{image_id}")
        assert response.headers["content-type"] == "image/jpeg"

    def test_invalid_image_returns_404(self, client):
        """Non-existent image ID should return 404."""
        response = client.get("/images/9999999")
        assert response.status_code == 404
