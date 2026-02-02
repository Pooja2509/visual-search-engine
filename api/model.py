"""
Model module — loads ResNet50 and extracts embeddings from images.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from api.config import (
    MODEL_PATH, IMAGE_SIZE, CROP_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, EMBEDDING_DIM
)


class EmbeddingExtractor:
    """
    Loads ResNet50 and extracts embeddings from images.

    Usage:
        extractor = EmbeddingExtractor()
        extractor.load()
        embedding = extractor.extract(image)
    """

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.is_loaded = False

    def load(self):
        """Load the model and preprocessing pipeline."""

        # --- STEP 1: Set device (CPU for Mac, GPU if available) ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # --- STEP 2: Create ResNet50 ---
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()    # remove classification layer

        # --- STEP 3: Try to load fine-tuned weights ---
        if os.path.exists(MODEL_PATH):
            print(f"Loading fine-tuned weights from {MODEL_PATH}...")
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded!")
        else:
            print(f"WARNING: {MODEL_PATH} not found.")
            print("Using baseline ResNet50 weights (not fine-tuned).")

        # --- STEP 4: Set to evaluation mode and move to device ---
        self.model = self.model.to(self.device)
        self.model.eval()

        # --- STEP 5: Define preprocessing pipeline ---
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),           # resize shorter side to 256
            transforms.CenterCrop(CROP_SIZE),        # crop center 224x224
            transforms.ToTensor(),                    # pixel values 0-255 → 0.0-1.0
            transforms.Normalize(
                mean=IMAGENET_MEAN,                   # center around 0
                std=IMAGENET_STD
            )
        ])

        self.is_loaded = True
        print("Embedding extractor ready!")

    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract embedding from a single PIL image.

        Args:
            image: PIL Image (any size, any mode)

        Returns:
            numpy array of shape (1, 2048), L2 normalized
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # --- STEP 1: Convert to RGB ---
        image = image.convert('RGB')

        # --- STEP 2: Preprocess ---
        img_tensor = self.transform(image)        # shape: (3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0)       # shape: (1, 3, 224, 224)
        img_tensor = img_tensor.to(self.device)    # move to CPU/GPU

        # --- STEP 3: Extract embedding ---
        with torch.no_grad():                      # don't compute gradients
            embedding = self.model(img_tensor)      # shape: (1, 2048)

        # --- STEP 4: Convert to numpy and L2 normalize ---
        embedding = embedding.cpu().numpy()         # GPU → CPU → numpy
        # L2 normalize: divide by length so vector has length 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding    # shape: (1, 2048)
