"""
Streamlit Frontend — Visual Search Engine UI.

Run with: streamlit run frontend/app.py
"""
import streamlit as st
import requests
from PIL import Image
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Search Engine",
    page_icon="🔍",
    layout="wide"
)

# --- API URL ---
# In Docker: reads from environment variable (http://api:8000)
# Locally: defaults to http://localhost:8000
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# --- Title and Description ---
st.title("🔍 Visual Search Engine")
st.markdown("Upload a fashion product image to find visually similar items.")
st.markdown("---")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=10)

# --- Check if API is running ---
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health["status"] == "healthy":
        st.sidebar.success(f"API Connected | {health['index_size']:,} products indexed")
    else:
        st.sidebar.error("API is not healthy")
except requests.exceptions.ConnectionError:
    st.sidebar.error("API is not running! Start it with: uvicorn api.main:app --port 8000")
    st.stop()

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a fashion product image (shirt, shoe, watch, etc.)"
)

# --- Main Search Logic ---
if uploaded_file is not None:

    # --- Show Query Image ---
    query_image = Image.open(uploaded_file)

    col_query, col_info = st.columns([1, 3])
    with col_query:
        st.subheader("Query Image")
        st.image(query_image, width=200)

    # --- Call API ---
    with st.spinner("Searching for similar products..."):
        # Reset file pointer to beginning (we already read it to display)
        uploaded_file.seek(0)

        # Send to API
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"top_k": top_k}

        try:
            response = requests.post(
                f"{API_URL}/search",
                files=files,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Search failed: {str(e)}")
            st.stop()

    # --- Show Search Info ---
    with col_info:
        st.subheader("Search Results")
        st.markdown(
            f"Found **{data['num_results']}** similar products "
            f"in **{data['search_time_ms']}ms**"
        )

    st.markdown("---")

    # --- Display Results in Grid ---
    results = data["results"]

    # 5 columns per row
    cols_per_row = 5

    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)

        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(results):
                break

            result = results[idx]

            with col:
                # Load result image from API
                try:
                    img_response = requests.get(
                        f"{API_URL}/images/{result['id']}",
                        timeout=5
                    )
                    result_image = Image.open(io.BytesIO(img_response.content))
                    st.image(result_image, use_container_width=True)
                except Exception:
                    st.write("Image not available")

                # Show product info
                score_pct = f"{result['score'] * 100:.1f}%"
                st.markdown(f"**Score: {score_pct}**")
                st.caption(f"{result['articleType']}")
                st.caption(f"{result.get('productDisplayName', 'Unknown')}")

else:
    # --- No image uploaded yet — show instructions ---
    st.info("👆 Upload an image above to start searching!")

    # --- Show example ---
    st.subheader("How it works")
    st.markdown("""
    1. **Upload** a fashion product image (shirt, shoe, watch, etc.)
    2. **AI model** (ResNet50 + Triplet Loss) extracts visual features
    3. **FAISS index** searches 43,916 products in milliseconds
    4. **Results** show the most visually similar products
    """)
