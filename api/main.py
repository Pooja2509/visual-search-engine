"""
FastAPI application — Visual Search API.

Endpoints:
    GET  /health     → check if server is running
    POST /search     → upload image, get similar products
    GET  /images/{id} → get product image by ID
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import time

from api.config import IMAGE_DIR, TOP_K
from api.model import EmbeddingExtractor
from api.search import SearchEngine

# --- Create FastAPI app ---
app = FastAPI(
    title="Visual Search API",
    description="Upload an image to find visually similar products",
    version="1.0.0"
)

# --- CORS middleware ---
# Allows the frontend (running on different port) to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins (for development)
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)

# --- Load model and search engine when server starts ---
extractor = EmbeddingExtractor()
search_engine = SearchEngine()


@app.on_event("startup")
def startup():
    """Runs once when the server starts."""
    print("\n" + "=" * 50)
    print("Starting Visual Search API...")
    print("=" * 50)
    extractor.load()
    search_engine.load()
    print("=" * 50)
    print("API is ready!")
    print("=" * 50 + "\n")


# ============================================================
# ENDPOINT 1: Health Check
# ============================================================
@app.get("/health")
def health_check():
    """Check if the server is running and models are loaded."""
    return {
        "status": "healthy",
        "model_loaded": extractor.is_loaded,
        "index_loaded": search_engine.is_loaded,
        "index_size": search_engine.index.ntotal if search_engine.is_loaded else 0,
    }


# ============================================================
# ENDPOINT 2: Visual Search
# ============================================================
@app.post("/search")
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = TOP_K
):
    """
    Upload an image and get visually similar products.

    - **file**: image file (JPEG, PNG, etc.)
    - **top_k**: number of results to return (default: 10)
    """
    start_time = time.time()

    # --- STEP 1: Validate the uploaded file ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )

    # --- STEP 2: Read the image ---
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {str(e)}"
        )

    # --- STEP 3: Extract embedding ---
    embedding = extractor.extract(image)

    # --- STEP 4: Search for similar products ---
    results = search_engine.search(embedding, top_k=top_k)

    # --- STEP 5: Return results ---
    elapsed = time.time() - start_time

    return {
        "query_filename": file.filename,
        "num_results": len(results),
        "search_time_ms": round(elapsed * 1000, 1),
        "results": results,
    }


# ============================================================
# ENDPOINT 3: Serve Product Images
# ============================================================
@app.get("/images/{image_id}")
def get_image(image_id: int):
    """Get a product image by its ID."""
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found"
        )

    return FileResponse(image_path)
