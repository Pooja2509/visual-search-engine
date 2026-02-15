# Architecture Deep Dive — Visual Search Engine

A detailed walkthrough of every component: theory, math, and code.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Extraction — ResNet50](#3-feature-extraction--resnet50)
4. [Metric Learning — Triplet Loss](#4-metric-learning--triplet-loss)
5. [Training Pipeline](#5-training-pipeline)
6. [FAISS Indexing](#6-faiss-indexing)
7. [Distance Math — L2 to Cosine Similarity](#7-distance-math--l2-to-cosine-similarity)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [API Architecture](#9-api-architecture)
10. [Frontend Architecture](#10-frontend-architecture)
11. [Docker Architecture](#11-docker-architecture)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [End-to-End Request Flow](#13-end-to-end-request-flow)

---

## 1. System Overview

### What does this system do?

Input: A fashion product image (shirt, shoe, watch, etc.)
Output: Top-K visually similar products from a catalog of 43,916 items

### High-level flow

```
User uploads image
       │
       ▼
┌──────────────┐
│  Preprocess   │  Resize to 256px, center crop to 224px, normalize
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ResNet50     │  Extract 2048-dimensional feature vector
│  (backbone)   │  Input: 224×224×3 image → Output: 1×2048 vector
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  L2 Normalize │  Make vector unit length (magnitude = 1.0)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  FAISS Search │  Compare against 43,916 pre-computed vectors
│  (nearest     │  Find K nearest neighbors by L2 distance
│   neighbor)   │  Convert L2 distance → cosine similarity score
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Metadata     │  Enrich results with product name, category, color
│  Enrichment   │  Return as JSON with similarity scores
└──────────────┘
```

---

## 2. Data Pipeline

### Dataset

Fashion Product Images (Small) from Kaggle:
- 44,441 product images (60×80 pixels)
- 143 article types (Shirts, Shoes, Watches, etc.)
- CSV metadata: id, articleType, masterCategory, baseColour, season, etc.

### Data Filtering

```python
# Remove categories with too few images
category_counts = styles['articleType'].value_counts()
valid_categories = category_counts[category_counts >= 100].index
filtered = styles[styles['articleType'].isin(valid_categories)]
```

Why ≥100 images?
- Triplet loss needs multiple images per category per batch
- Categories with <100 images have too few for meaningful training
- After filtering: 43,916 images across ~40 categories

### Stratified Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train, temp = train_test_split(filtered, test_size=0.3,
                                stratify=filtered['articleType'],
                                random_state=42)
val, test = train_test_split(temp, test_size=0.5,
                              stratify=temp['articleType'],
                              random_state=42)
```

Why stratified?
- `stratify=filtered['articleType']` ensures each category has the same
  proportion in train/val/test
- Without stratification: rare categories might end up entirely in test
  → model never trains on them
- Example: If "Kurtas" is 2% of data → 2% in train, 2% in val, 2% in test

### Image Preprocessing

```python
transforms.Compose([
    transforms.Resize(256),         # Step 1: Resize shorter side to 256px
    transforms.CenterCrop(224),     # Step 2: Crop center 224×224
    transforms.ToTensor(),          # Step 3: Convert to tensor [0, 1]
    transforms.Normalize(           # Step 4: Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Step-by-step for a 60×80 image:

```
Original:     60×80×3 (H×W×C, uint8, values 0-255)
     │
Resize(256):  256×341×3 (shorter side becomes 256, aspect ratio preserved)
     │         WHY? ResNet expects ~224px input. We resize to 256 first,
     │         then crop to 224 (gives a small border for cropping)
     │
CenterCrop:   224×224×3 (take center square)
     │         WHY? ResNet expects square 224×224 input.
     │         Center crop preserves the main product (usually centered)
     │
ToTensor:     3×224×224 (float32, values 0.0-1.0)
     │         WHY? PyTorch expects (C, H, W) format, not (H, W, C)
     │         Also converts uint8 [0,255] → float32 [0.0, 1.0]
     │
Normalize:    3×224×224 (float32, values roughly -2.0 to 2.0)
              WHY? ImageNet pretrained models expect this normalization
              Formula: pixel = (pixel - mean) / std
              Example: Red channel pixel 0.5 → (0.5 - 0.485) / 0.229 = 0.065
```

Why ImageNet mean/std?
- ResNet50 was pretrained on ImageNet with these exact values
- The model's weights "expect" input in this range
- Using different normalization → model outputs are garbage

---

## 3. Feature Extraction — ResNet50

### What is ResNet50?

A 50-layer deep convolutional neural network that won ImageNet 2015.

```
ResNet50 Architecture:
  Input: 3×224×224 (RGB image)
    │
    ▼
  conv1: 7×7 conv, 64 filters, stride 2 → 64×112×112
    │
    ▼
  maxpool: 3×3, stride 2 → 64×56×56
    │
    ▼
  layer1: 3 bottleneck blocks → 256×56×56      (edges, textures)
    │
    ▼
  layer2: 4 bottleneck blocks → 512×28×28      (patterns, shapes)
    │
    ▼
  layer3: 6 bottleneck blocks → 1024×14×14     (parts: sleeves, collars)
    │
    ▼
  layer4: 3 bottleneck blocks → 2048×7×7       (objects: shirts, shoes)
    │
    ▼
  avgpool: adaptive average pooling → 2048×1×1  (global features)
    │
    ▼
  flatten: → 2048                               (embedding vector!)
    │
    ▼
  fc: 2048 → 1000 (ImageNet classes)            (WE REMOVE THIS)
```

### Bottleneck Block (the building block of ResNet)

```
Input: x (256 channels)
  │
  ├──────────────────────────┐
  │                          │ (skip/residual connection)
  ▼                          │
  1×1 conv (256 → 64)       │  ← reduce channels (cheaper computation)
  │                          │
  3×3 conv (64 → 64)        │  ← actual spatial convolution
  │                          │
  1×1 conv (64 → 256)       │  ← restore channels
  │                          │
  ▼                          │
  + ◄────────────────────────┘  ← ADD input to output (residual!)
  │
  ReLU
  │
  ▼
Output: F(x) + x
```

Why residual connections?
- Without residual: output = F(x) → deeper networks are HARDER to train
- With residual: output = F(x) + x → network only needs to learn the DIFFERENCE
- If the layer is useless, F(x) = 0, output = x (identity, no harm done)
- This is why ResNet can be 50, 101, or 152 layers deep without degrading

### Removing the Classification Head

```python
# Standard ResNet50 (for classification)
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model.fc = Linear(2048 → 1000)  ← classifies into 1000 ImageNet classes

# Our ResNet50 (for embeddings)
model.fc = nn.Identity()
# model.fc = Identity()  ← passes 2048-dim vector through unchanged
```

Why remove fc?
- The fc layer maps 2048-dim → 1000 classes (cat, dog, car, etc.)
- We don't want classification — we want the 2048-dim FEATURES
- These features capture visual characteristics: shape, color, texture, pattern
- nn.Identity() is a no-op layer: input passes through unchanged
- Now model(image) returns a 2048-dim vector instead of 1000 class probabilities

### Parameter Count

```
Layer     | Parameters | Trainable in our setup
----------|------------|----------------------
conv1     |     9,408  | ❄️ Frozen
layer1    |   215,808  | ❄️ Frozen
layer2    |   1,219,584 | ❄️ Frozen
layer3    |   7,098,368 | ❄️ Frozen
layer4    |  14,964,736 | 🔥 TRAINABLE
fc (removed)| 2,049,000 | Removed
----------|------------|
Total     |  25,557,032|
Trainable |  14,964,736| (58.6%)
```

Why freeze layers 1-3?
- Early layers learn UNIVERSAL features (edges, corners, textures)
- These features work for ANY image domain (faces, cars, fashion)
- They don't need to change for fashion products
- Layer4 learns HIGH-LEVEL features (specific to the domain)
- Training only layer4: faster training, less overfitting, stable convergence

---

## 4. Metric Learning — Triplet Loss

### The Goal

We want an embedding space where:
- Images of the SAME category → close together
- Images of DIFFERENT categories → far apart

```
Before training:                  After training:
  (random embedding space)          (organized embedding space)

  👟 👗 ⌚ 👔                      👟👟👟   👗👗👗
  👗 👟 👔 ⌚                          ⌚⌚⌚   👔👔👔
  ⌚ 👔 👟 👗
  👔 ⌚ 👗 👟                    (same types cluster together!)
```

### Triplet Loss Formula

For each triplet (anchor, positive, negative):

```
L = max(0, d(a, p) - d(a, n) + margin)

Where:
  a = anchor image embedding      (e.g., a red shirt)
  p = positive image embedding    (e.g., another red shirt — same category)
  n = negative image embedding    (e.g., blue shoes — different category)
  d(x, y) = Euclidean distance between x and y
  margin = 0.3 (minimum gap between positive and negative distances)
```

### Visual explanation

```
Case 1: EASY triplet (loss = 0)
  d(a,p) = 0.2    d(a,n) = 0.8    margin = 0.3
  L = max(0, 0.2 - 0.8 + 0.3) = max(0, -0.3) = 0
  ✅ Negative is already far away. Model learns nothing (loss is 0).

Case 2: HARD triplet (loss > 0)
  d(a,p) = 0.5    d(a,n) = 0.4    margin = 0.3
  L = max(0, 0.5 - 0.4 + 0.3) = max(0, 0.4) = 0.4
  ❌ Negative is CLOSER than positive! Model must fix this.

Case 3: SEMI-HARD triplet (small loss)
  d(a,p) = 0.3    d(a,n) = 0.5    margin = 0.3
  L = max(0, 0.3 - 0.5 + 0.3) = max(0, 0.1) = 0.1
  ⚠️ Negative is farther, but not by enough (< margin).
```

### Batch Hard Mining

Instead of randomly selecting triplets, we mine the HARDEST ones from each batch:

```python
def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    # 1. Compute ALL pairwise distances in the batch
    #    For batch size 64: compute 64×64 = 4,096 distances
    distances = torch.cdist(embeddings, embeddings, p=2)

    # 2. For each anchor, find:
    #    - Hardest positive: FARTHEST image with SAME label
    #    - Hardest negative: CLOSEST image with DIFFERENT label

    for each anchor i:
        # All images with same label as anchor
        positive_distances = distances[i][labels == labels[i]]
        hardest_positive = max(positive_distances)  # farthest same-class

        # All images with different label
        negative_distances = distances[i][labels != labels[i]]
        hardest_negative = min(negative_distances)  # closest different-class

        # Triplet loss for this anchor
        loss_i = max(0, hardest_positive - hardest_negative + margin)

    # 3. Average over all anchors
    return mean(all loss_i values)
```

Why batch hard?
- Random triplets: 90%+ are easy (loss = 0) → model learns nothing
- Hard mining: Every triplet is informative → faster, better convergence
- We tracked "active triplets" per batch: started at 53/64, dropped to 19/64
  → model was solving more triplets correctly as training progressed

### Balanced Batch Sampling

```python
class BalancedBatchSampler:
    # P = 16 classes per batch
    # K = 4 images per class
    # Batch size = P × K = 64
```

Why balanced?
- Random sampling: A batch might have 60 shirts and 4 shoes
  → only 1 negative class → poor triplet diversity
- Balanced: 16 different classes × 4 images each
  → 16 positive pairs per class, 48 negative classes
  → rich triplet mining

```
Example batch (64 images):
  Shirts:     img1, img2, img3, img4      (4 positives for each other)
  Shoes:      img5, img6, img7, img8      (4 positives for each other)
  Watches:    img9, img10, img11, img12
  Bags:       img13, img14, img15, img16
  ... (12 more classes, 4 each)

  Total: 16 classes × 4 images = 64 per batch

  For anchor=img1 (Shirt):
    Hardest positive = farthest among {img2, img3, img4}
    Hardest negative = closest among {img5, img6, ..., img64}
```

---

## 5. Training Pipeline

### Training Configuration

```python
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
```

- `Adam`: Adaptive learning rate optimizer (adjusts per-parameter)
- `lr=1e-5`: Small learning rate because we're fine-tuning (not training from scratch)
  - Training from scratch: lr=1e-3 (1000× larger)
  - Fine-tuning: lr=1e-5 (gentle updates to preserve pretrained knowledge)
- `weight_decay=1e-4`: L2 regularization to prevent overfitting
  - Adds penalty: loss = triplet_loss + 1e-4 × sum(weights²)
  - Keeps weights small → simpler model → less overfitting

### Training Loop

```python
for epoch in range(max_epochs):  # max_epochs = 30

    # === TRAINING PHASE ===
    model.train()  # Enable dropout, batch norm in training mode
    for images, labels in train_loader:

        # Forward pass
        embeddings = model(images)           # (64, 2048)
        embeddings = F.normalize(embeddings) # L2 normalize

        # Compute loss
        loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights

    # === VALIDATION PHASE ===
    model.eval()  # Disable dropout, batch norm in eval mode
    with torch.no_grad():  # Don't compute gradients (saves memory)
        for images, labels in val_loader:
            embeddings = model(images)
            embeddings = F.normalize(embeddings)
            val_loss = batch_hard_triplet_loss(embeddings, labels)

    # === EARLY STOPPING ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best
    else:
        patience_counter += 1
        if patience_counter >= 3:  # patience = 3
            model.load_state_dict(torch.load('best_model.pth'))
            break  # Stop training
```

### Early Stopping — Why?

```
Epoch 1: val_loss = 0.45   ← improving
Epoch 2: val_loss = 0.38   ← improving
Epoch 3: val_loss = 0.33   ← improving
Epoch 4: val_loss = 0.30   ← improving
Epoch 5: val_loss = 0.29   ← BEST! Save checkpoint ✓
Epoch 6: val_loss = 0.31   ← worse (patience: 1/3)
Epoch 7: val_loss = 0.32   ← worse (patience: 2/3)
Epoch 8: val_loss = 0.33   ← worse (patience: 3/3) → STOP!

Load checkpoint from Epoch 5 (best model)
```

Without early stopping:
- Training loss keeps decreasing (model memorizes training data)
- Validation loss starts INCREASING (model is overfitting)
- You end up with a worse model

### Training Results

```
Epochs trained: 11 (early stopped at epoch 8)
Best validation loss: 0.2939 (epoch 5)
Active triplets: 53/64 → 19/64 (model solved more triplets over time)
Training time: ~45 minutes on Google Colab GPU
```

---

## 6. FAISS Indexing

### What is FAISS?

Facebook AI Similarity Search — a library optimized for finding nearest
neighbors in high-dimensional spaces.

### Why not just use numpy?

```python
# Naive approach: compute distance to ALL 43,916 items
distances = np.linalg.norm(all_embeddings - query_embedding, axis=1)
top_k = np.argsort(distances)[:10]

# This works! But:
# - 43,916 × 2048 multiplications per query
# - NumPy: ~20ms per query
# - FAISS: ~2ms per query (10× faster)
# - At 1M items: NumPy ~500ms, FAISS ~5ms (100× faster)
```

FAISS is faster because:
- Uses SIMD instructions (processes 4-8 floats simultaneously)
- Optimized memory access patterns (cache-friendly)
- Optional: approximate algorithms for even faster search

### Index Types We Benchmarked

```
1. Flat Index (IndexFlatL2)
   - Brute force: compare query against every vector
   - 100% recall (perfect accuracy)
   - O(n×d) per query
   - Our results: 2,183 queries/second

2. IVF Index (IndexIVFFlat)
   - Partition vectors into clusters using K-means
   - At query time: only search the nearest clusters
   - nprobe parameter controls accuracy vs speed
   - Our results at nprobe=10: 100% recall, 1,581 q/s

3. IVF+PQ Index (IndexIVFPQ)
   - IVF clustering + Product Quantization compression
   - Compress each vector from 8KB to 64 bytes (128× smaller!)
   - Our results: 62-70% recall (too lossy for us)
```

### Building the Flat Index

```python
import faiss
import numpy as np

# Load embeddings: shape (43916, 2048), float32
embeddings = np.load('finetuned_embeddings.npy').astype('float32')

# L2 normalize (important: makes L2 distance equivalent to cosine)
faiss.normalize_L2(embeddings)

# Create index
dimension = 2048
index = faiss.IndexFlatL2(dimension)

# Add all vectors
index.add(embeddings)

# Search
query = extract_embedding(uploaded_image)  # shape (1, 2048)
faiss.normalize_L2(query)
distances, indices = index.search(query, k=10)

# distances: shape (1, 10) — L2 distances to 10 nearest neighbors
# indices: shape (1, 10) — positions of those neighbors in the index
```

### How IVF Works (for understanding)

```
Step 1: TRAINING — cluster vectors using K-means

  43,916 vectors → K-means with 256 clusters
  Each cluster has a centroid (center point)

  Cluster 0: [vectors about shoes]      centroid_0
  Cluster 1: [vectors about shirts]     centroid_1
  Cluster 2: [vectors about watches]    centroid_2
  ...
  Cluster 255: [vectors about bags]     centroid_255

Step 2: SEARCH — only search nearby clusters

  Query: new shoe image

  1. Find nearest centroids to query: cluster 0, cluster 47, cluster 183
     (these are the nprobe=3 nearest clusters)

  2. Only search vectors in those 3 clusters
     Instead of 43,916 comparisons → maybe 500 comparisons

  3. Return top-K from those 500

  Tradeoff: might miss a relevant result in cluster 200
            (but unlikely if nprobe is large enough)
```

### How Product Quantization Works (for understanding)

```
Original vector: 2048 floats = 8,192 bytes (8KB)

PQ splits it into 64 sub-vectors of 32 dimensions each:
  [sub_0(32d)] [sub_1(32d)] [sub_2(32d)] ... [sub_63(32d)]

For each sub-vector position, K-means creates 256 cluster centers (codebook):
  Sub-vector 0: 256 possible codes → 8 bits to select one
  Sub-vector 1: 256 possible codes → 8 bits to select one
  ...
  Sub-vector 63: 256 possible codes → 8 bits to select one

Compressed vector: 64 × 8 bits = 64 bytes (instead of 8,192 bytes!)
  [code_0] [code_1] [code_2] ... [code_63]

Compression ratio: 8192 / 64 = 128×

Distance computation:
  Instead of computing actual L2 distance (expensive),
  use precomputed distances between query sub-vectors and codebook entries
  This is an APPROXIMATION → explains why recall drops to 62-70%
```

### Why We Chose Flat Index

```
Dataset size: 43,916 items
Flat index memory: 43,916 × 2048 × 4 bytes = 343 MB
Search time: ~2ms per query

At this scale:
  ✅ 343 MB fits in RAM easily
  ✅ 2ms is fast enough for real-time search
  ✅ 100% recall — no accuracy loss
  ✅ Simplest to implement and maintain

When to switch to IVF/PQ:
  100K-1M items → IVF (index > 800MB, search > 10ms)
  1M-10M items → IVF+PQ (index > 8GB, won't fit in RAM)
```

---

## 7. Distance Math — L2 to Cosine Similarity

### The Key Identity

For L2-normalized vectors (||a|| = ||b|| = 1):

```
L2²(a, b) = ||a - b||²
           = ||a||² + ||b||² - 2(a · b)
           = 1 + 1 - 2(a · b)
           = 2 - 2·cos(θ)
           = 2(1 - cos(θ))

Therefore:
  cos(θ) = 1 - L2²/2

Where:
  cos(θ) = cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
  L2² = squared L2 distance (0.0 = identical, 4.0 = opposite)
```

### Step-by-step proof

```
||a - b||² = (a - b) · (a - b)                    [definition of L2²]
           = a·a - 2(a·b) + b·b                    [expand dot product]
           = ||a||² - 2(a·b) + ||b||²              [definition of norm]
           = 1 - 2(a·b) + 1                        [both are unit vectors]
           = 2 - 2(a·b)                            [simplify]
           = 2 - 2·cos(θ)                          [a·b = ||a||·||b||·cos(θ) = cos(θ)]
           = 2(1 - cos(θ))                         [factor out 2]

Solving for cos(θ):
  L2² = 2(1 - cos(θ))
  L2²/2 = 1 - cos(θ)
  cos(θ) = 1 - L2²/2
```

### In our code

```python
# FAISS returns L2 distances (not squared, just L2)
# But actually FAISS IndexFlatL2 returns SQUARED L2 distances!
distance = float(distances[0][rank])   # This is L2² (squared)

# Convert to cosine similarity
cosine_sim = 1.0 - distance / 2.0

# Examples:
#   distance = 0.0   → cosine_sim = 1.0    (identical images)
#   distance = 0.05  → cosine_sim = 0.975  (very similar)
#   distance = 1.0   → cosine_sim = 0.5    (somewhat similar)
#   distance = 2.0   → cosine_sim = 0.0    (orthogonal/unrelated)
#   distance = 4.0   → cosine_sim = -1.0   (opposite — very rare)
```

Why show cosine similarity instead of L2 distance?
- "99.7% similar" is intuitive for users
- "L2 distance 0.0548" means nothing to non-ML people
- Cosine similarity is bounded [−1, 1], easy to interpret

---

## 8. Evaluation Metrics

### Precision@K

```
P@K = (number of relevant results in top K) / K

Example: Search for "Shirts", K=5
  Result 1: Shirt ✅
  Result 2: Shirt ✅
  Result 3: Blouse ❌
  Result 4: Shirt ✅
  Result 5: Pants ❌

  P@5 = 3/5 = 0.6 (60%)

Interpretation: "60% of the top 5 results were relevant"
```

### Mean Reciprocal Rank (MRR)

```
RR = 1 / (rank of first relevant result)

Example 1: First result is relevant
  Results: Shirt✅, Pants, Shoes, Shirt, Bag
  RR = 1/1 = 1.0

Example 2: Third result is first relevant
  Results: Pants, Shoes, Shirt✅, Shirt, Bag
  RR = 1/3 = 0.333

MRR = average of RR across all queries

Interpretation: "How quickly do users find a relevant result?"
  MRR = 0.90 means on average, the first relevant result is at position 1.1
```

### Mean Average Precision (mAP@K)

```
AP@K considers the ORDER of relevant results:

Example: Search for "Shirts", K=5
  Result 1: Shirt ✅  → P@1 = 1/1 = 1.0
  Result 2: Shirt ✅  → P@2 = 2/2 = 1.0
  Result 3: Blouse ❌  → (skip, not relevant)
  Result 4: Shirt ✅  → P@4 = 3/4 = 0.75
  Result 5: Pants ❌  → (skip, not relevant)

  AP@5 = (1.0 + 1.0 + 0.75) / 3 = 0.917

  Only precision at positions with relevant results counts.

  Compare two orderings:
    [✅ ✅ ❌ ✅ ❌]  AP = (1.0 + 1.0 + 0.75) / 3 = 0.917   ← good
    [❌ ❌ ✅ ✅ ✅]  AP = (0.33 + 0.5 + 0.6) / 3 = 0.477   ← bad

  Same number of relevant results (3), but first ordering is much better
  because relevant results appear earlier.

mAP = average of AP across all queries

Interpretation: "How good is the ranking quality?"
```

### Our Results

```
Evaluated on 6,588 test queries:

  Metric          | Baseline (ImageNet) | Fine-tuned (Triplet)
  ─────────────────────────────────────────────────────────────
  Precision@5     |     72.0%           |     82.61%  (+10.6%)
  MRR             |     ~80%            |     90.22%
  mAP@10          |     ~76%            |     86.79%
  Median latency  |      —              |     6.81ms

Best categories:  Sunglasses (100%), Watches (96.1%)
Worst categories: Tops (67.4%), Casual Shoes (71.2%)

Why Tops are hardest:
  "Tops" is a broad category — includes t-shirts, blouses, tunics
  A red t-shirt and a white blouse are both "Tops" but look very different
  The model correctly puts them far apart, but our metric calls it a miss
```

---

## 9. API Architecture

### Startup Flow

```python
@app.on_event("startup")
def startup():
    extractor.load()        # Load ResNet50 + weights (~10 seconds)
    search_engine.load()    # Load FAISS index + metadata (~5 seconds)
```

Why load at startup (not per-request)?
- Loading model: ~10 seconds (read 90MB from disk, build GPU graph)
- Loading index: ~5 seconds (read 343MB from disk)
- If we loaded per-request: every search takes 15+ seconds
- Loading once at startup: searches take ~50ms

### Request Flow for POST /search

```
Client sends: POST /search (multipart form with image file)
  │
  ▼
1. Validate file type
   if not file.content_type.startswith("image/"):
       raise HTTPException(400, "File must be an image")
  │
  ▼
2. Read image bytes into memory
   image_bytes = await file.read()     # async! doesn't block other requests
   image = Image.open(io.BytesIO(image_bytes))
  │
  ▼
3. Extract embedding
   embedding = extractor.extract(image)
   # Internally:
   #   a. Resize, crop, normalize (transform pipeline)
   #   b. Forward pass through ResNet50
   #   c. L2 normalize the output
   #   d. Returns numpy array shape (1, 2048)
  │
  ▼
4. Search FAISS index
   results = search_engine.search(embedding, top_k=10)
   # Internally:
   #   a. distances, indices = index.search(embedding, k=11)  [k+1 to skip self]
   #   b. Convert L2 distance → cosine similarity
   #   c. Look up metadata for each result ID
   #   d. Returns list of dicts with id, score, articleType, etc.
  │
  ▼
5. Return JSON response
   {
     "query_filename": "shirt.jpg",
     "num_results": 10,
     "search_time_ms": 47.3,
     "results": [
       {"id": 15970, "score": 0.997, "articleType": "Shirts", ...},
       {"id": 39386, "score": 0.981, "articleType": "Shirts", ...},
       ...
     ]
   }
```

### Why FastAPI (not Flask)?

```python
# FastAPI — async file upload
async def visual_search(file: UploadFile = File(...)):
    image_bytes = await file.read()    # Non-blocking!

# While this request reads the file, FastAPI can handle OTHER requests
# Flask would block on file.read() — one request at a time

# FastAPI also gives us:
#   - Auto-generated docs at /docs (Swagger UI)
#   - Type validation (UploadFile, int, etc.)
#   - Auto error responses (422 for missing fields)
```

---

## 10. Frontend Architecture

### Streamlit Flow

```python
# 1. User uploads image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 2. If file uploaded, call API
if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{API_URL}/search", files=files, params={"top_k": top_k})
    data = response.json()

# 3. Display results in 5-column grid
    cols_per_row = 5
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            result = results[idx]
            with col:
                img = requests.get(f"{API_URL}/images/{result['id']}")
                st.image(Image.open(io.BytesIO(img.content)))
                st.markdown(f"**Score: {result['score']*100:.1f}%**")
```

### Grid Layout Logic

```
10 results, 5 columns per row → 2 rows

Row 1 (row_start=0): results[0] results[1] results[2] results[3] results[4]
Row 2 (row_start=5): results[5] results[6] results[7] results[8] results[9]

The nested loop:
  range(0, 10, 5) → [0, 5]           (row starts)
  enumerate(cols) → [(0,col), (1,col), ..., (4,col)]   (columns)
  idx = row_start + i:
    Row 1: 0+0=0, 0+1=1, 0+2=2, 0+3=3, 0+4=4
    Row 2: 5+0=5, 5+1=6, 5+2=7, 5+3=8, 5+4=9
```

---

## 11. Docker Architecture

### Container Architecture

```
┌─────────────────────────────────────────────────────┐
│              Docker Compose Network                   │
│                                                       │
│  ┌───────────────────┐    ┌───────────────────────┐  │
│  │ visual-search-api  │    │ visual-search-frontend │  │
│  │                    │    │                         │  │
│  │ CMD: uvicorn       │    │ CMD: streamlit run      │  │
│  │ Port: 8000         │◄───│ Port: 8501              │  │
│  │                    │    │                         │  │
│  │ Contains:          │    │ API_URL=http://api:8000 │  │
│  │  - ResNet50 model  │    │                         │  │
│  │  - FAISS index     │    │ Same image, different   │  │
│  │  - 44K images      │    │ CMD (streamlit vs       │  │
│  │  - Metadata CSV    │    │ uvicorn)                │  │
│  └───────────────────┘    └───────────────────────┘  │
│         ▲                        ▲                    │
│    host:8000                host:8501                  │
└─────────────────────────────────────────────────────┘
```

### Dockerfile Layer Strategy

```
Layer 1: FROM python:3.11-slim           (~150MB, changes: never)
Layer 2: RUN apt-get install curl         (~5MB, changes: never)
Layer 3: COPY requirements.txt            (~1KB, changes: rarely)
Layer 4: RUN pip install                  (~1.5GB, changes: rarely)
Layer 5: COPY api/, frontend/             (~20KB, changes: often)
Layer 6: COPY models/, data/, indexing/   (~1.4GB, changes: rarely)

Why this order?
  Frequently-changing layers at the BOTTOM
  Rarely-changing layers at the TOP

  If you change api/main.py:
    Layers 1-4: CACHED ✓ (instant)
    Layer 5: REBUILT (1 second)
    Layer 6: REBUILT (must rebuild all after changed layer)

  If you change requirements.txt:
    Layers 1-2: CACHED ✓
    Layers 3-6: ALL REBUILT (5+ minutes)

  Code changes (frequent) → fast rebuild
  Dependency changes (rare) → slow rebuild (acceptable)
```

### Key Docker Decisions

```
CPU-only PyTorch:  200MB instead of 2.5GB (saves 2.3GB)
.dockerignore:     Excludes baseline data (saves ~700MB)
Single Dockerfile: Both services share one image (simpler maintenance)
0.0.0.0 binding:   Allows cross-container and host access
start_period: 60s: Grace period for model loading before health checks
```

---

## 12. CI/CD Pipeline

### Pipeline Flow

```
git push to main
       │
       ▼
┌─────────────────┐
│  Job 1: LINT    │  flake8 checks:
│  ubuntu-latest  │    - Syntax errors
│  ~30 seconds    │    - Unused imports
│                 │    - Style violations
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│  Job 2: TEST    │  pytest runs:
│  ubuntu-latest  │    - 16 config tests
│  ~2 minutes     │    - (model/API tests need data files)
│                 │    - Pip cache for faster installs
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│  Job 3: DOCKER  │  Verify:
│  ubuntu-latest  │    - Dockerfile exists
│  ~1 minute      │    - docker-compose.yml exists
│  (main only)    │    - Image available on Docker Hub
└─────────────────┘
```

### What Each Job Catches

```
Lint catches:
  - if x = 5:        (= instead of ==)
  - import os         (unused import)
  - def foo():pass    (missing whitespace)

Test catches:
  - EMBEDDING_DIM changed to 1024 (tests expect 2048)
  - CROP_SIZE > IMAGE_SIZE (invalid config)
  - Missing config variables

Docker catches:
  - Deleted Dockerfile accidentally
  - Broken docker-compose.yml syntax
```

---

## 13. End-to-End Request Flow

### Complete path of a single search request

```
1. USER opens browser → localhost:8501

2. STREAMLIT renders the page
   → Shows file uploader widget
   → Calls GET localhost:8000/health (checks API is running)

3. USER uploads "red_shirt.jpg"

4. STREAMLIT receives the file
   → Displays query image preview
   → Shows spinner "Searching..."

5. STREAMLIT → API: POST localhost:8000/search
   Body: multipart form data with image bytes
   Params: ?top_k=10

6. FASTAPI receives request
   → Validates content_type starts with "image/"
   → Reads file bytes asynchronously

7. PILLOW opens image bytes → PIL Image object

8. EMBEDDING EXTRACTOR processes image:
   a. Resize(256):     60×80 → 256×341
   b. CenterCrop(224): 256×341 → 224×224
   c. ToTensor:        224×224×3 (uint8) → 3×224×224 (float32)
   d. Normalize:       Apply ImageNet mean/std
   e. Unsqueeze:       3×224×224 → 1×3×224×224 (add batch dimension)
   f. model(tensor):   Forward pass through ResNet50
                        conv1 → layer1 → layer2 → layer3 → layer4 → avgpool
                        Output: 1×2048
   g. L2 normalize:    embedding / ||embedding||
   h. Return:          numpy array shape (1, 2048)

9. FAISS SEARCH:
   a. index.search(query, k=11)   # k+1 to handle self-match
   b. Returns: distances (1×11), indices (1×11)
   c. For each result:
      - Skip if distance ≈ 0 (self-match)
      - Convert: cosine_sim = 1 - distance/2
      - Look up metadata: id_to_meta[image_id]
      - Build result dict: {id, score, articleType, ...}
   d. Return top 10 results

10. FASTAPI returns JSON:
    {
      "query_filename": "red_shirt.jpg",
      "num_results": 10,
      "search_time_ms": 47.3,
      "results": [...]
    }

11. STREAMLIT receives JSON
    → For each result:
       → GET localhost:8000/images/{id}
       → Display image in grid column
       → Show score, category, product name

12. USER sees 10 similar products with scores!

Total time: ~50ms (model) + ~2ms (FAISS) + ~10ms (network) ≈ 62ms
```
