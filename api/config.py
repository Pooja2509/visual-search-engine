"""
Configuration file — all paths and settings in one place.
"""
import os

# Fix OpenMP conflict between PyTorch and FAISS on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base directory (parent of api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "finetuned_embeddings.npy")
IDS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "finetuned_ids.npy")
METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "filtered_styles.csv")
INDEX_PATH = os.path.join(BASE_DIR, "indexing", "flat_index.faiss")
MODEL_PATH = os.path.join(BASE_DIR, "models", "triplet_resnet50.pth")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")

# Search settings
TOP_K = 10              # number of results to return
EMBEDDING_DIM = 2048    # dimension of embeddings

# Image preprocessing settings (must match training)
IMAGE_SIZE = 256        # resize shorter side to this
CROP_SIZE = 224         # center crop to this
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
