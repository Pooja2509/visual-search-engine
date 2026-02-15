# Design Decisions — Visual Search Engine

This document explains the reasoning behind every major technical decision in this project.

---

## 1. Why ResNet50 (not ResNet18, ResNet152, or ViT)?

**Choice**: ResNet50

**Alternatives considered**:
- ResNet18: Faster but lower quality embeddings (512-dim vs 2048-dim)
- ResNet152: Better accuracy but 3x slower inference, diminishing returns
- ViT (Vision Transformer): State-of-the-art but much slower, larger model
- EfficientNet: Good accuracy/speed but less proven for retrieval tasks

**Reasoning**:
- ResNet50 is the industry standard for visual embedding extraction
- 2048-dim embeddings capture rich visual features
- Pretrained ImageNet weights give strong baseline without training
- Inference time ~50ms on CPU — fast enough for real-time search
- Well-supported by PyTorch, extensive documentation

**Tradeoff**: We sacrificed potential accuracy gains from larger models (ResNet152, ViT) in exchange for faster inference and simpler deployment.

---

## 2. Why Triplet Loss with Batch Hard Mining (not Contrastive Loss, ArcFace)?

**Choice**: Triplet Loss with batch hard mining

**Alternatives considered**:
- Contrastive Loss: Only uses pairs (positive/negative), less information per batch
- ArcFace/CosFace: Angular margin losses, state-of-the-art for face recognition
- Cross-entropy: Classification-based, doesn't directly optimize for retrieval

**Reasoning**:
- Triplet loss directly optimizes the embedding space for retrieval
- Batch hard mining selects the hardest examples in each batch automatically
- No need for explicit pair/triplet construction (mining happens at batch level)
- Margin parameter (0.3) controls how far apart different classes should be
- Well-proven for image retrieval in research (Google, Pinterest)

**Tradeoff**: Triplet loss can be unstable early in training (collapsing embeddings). We mitigated this by using pretrained weights and freezing early layers.

---

## 3. Why Freeze Layers 1-3, Train Only Layer4?

**Choice**: Freeze conv1 + layer1 + layer2 + layer3, train layer4 + fc

**Alternatives considered**:
- Train all layers: Full fine-tuning, risk of overfitting on small dataset
- Freeze all, train only fc: Too restrictive, limited adaptation
- Gradual unfreezing: Complex learning rate scheduling

**Reasoning**:
- Early layers (1-3) learn universal features: edges, textures, shapes
- These features transfer well across domains (ImageNet → fashion)
- Layer4 learns high-level, domain-specific features
- Training layer4 adapts the model to fashion-specific patterns
- 58.6% of parameters are in layer4 — significant capacity to learn
- Prevents catastrophic forgetting of useful pretrained features

**Tradeoff**: We sacrificed full model flexibility for training stability and reduced overfitting risk on our 44K image dataset.

---

## 4. Why FAISS Flat Index (not IVF, HNSW, Annoy)?

**Choice**: FAISS IndexFlatL2 (exact brute-force search)

**Alternatives considered**:
- FAISS IVF: Clustered search, faster but approximate (tested: 100% recall at nprobe=10)
- FAISS IVF+PQ: Compressed, 128x smaller but only 62-70% recall
- HNSW (hnswlib): Graph-based ANN, good recall but complex tuning
- Annoy (Spotify): Tree-based ANN, simpler but slower for high-dim

**Reasoning**:
- With only 43,916 items, exact search is fast enough (~2ms per query)
- Flat index guarantees 100% recall — no approximation errors
- Simplest to implement and maintain
- No training or parameter tuning needed
- FAISS is optimized for L2 distance on CPU (SIMD instructions)

**At what scale would we change?**
- 100K-1M items: Switch to IVF with nprobe=10-50
- 1M-10M items: Use IVF+PQ with OPQ rotation
- 10M+ items: Consider HNSW or distributed search

**Tradeoff**: We chose simplicity and perfect recall over search speed. At our scale (44K), the speed difference is negligible (2ms flat vs 1.5ms IVF).

---

## 5. Why FastAPI (not Flask, Django)?

**Choice**: FastAPI

**Alternatives considered**:
- Flask: Simpler but synchronous, no built-in validation
- Django: Full framework, too heavy for an API-only service
- Express.js: JavaScript, would need separate model serving

**Reasoning**:
- Async support: File uploads are I/O-bound, async handles concurrent requests better
- Auto-generated API docs (Swagger UI at /docs)
- Type validation with Pydantic: Catches errors before they reach the model
- UploadFile handling: Built-in multipart file upload support
- Lightweight: Only the features we need, no ORM/admin/templates
- Growing industry standard for ML APIs (used at Netflix, Uber, Microsoft)

**Tradeoff**: FastAPI has a steeper learning curve than Flask, but the benefits (async, validation, docs) outweigh the complexity for production use.

---

## 6. Why Streamlit (not React, Vue, Gradio)?

**Choice**: Streamlit

**Alternatives considered**:
- React: More customizable but requires JavaScript knowledge + separate build
- Gradio: Simpler than Streamlit but less customizable layout
- Flask templates: Old-school, more work for less result

**Reasoning**:
- Pure Python: No JavaScript, HTML, or CSS needed
- Rapid prototyping: Built the entire UI in ~120 lines of Python
- Built-in components: file uploader, image display, columns, sidebar
- Hot reload: Changes appear instantly during development
- Good enough for demos and portfolio projects

**Tradeoff**: Streamlit is not suitable for production web apps (no routing, limited state management). For a production system, we would use React + FastAPI.

---

## 7. Why Docker + Docker Compose (not just pip install)?

**Choice**: Docker containerization with docker-compose

**Alternatives considered**:
- Virtual environment only: Works locally but "works on my machine" problem
- Conda: Better for ML dependencies but not standard for deployment
- Kubernetes: Overkill for a single-service application

**Reasoning**:
- Reproducibility: Same image runs identically everywhere
- Isolation: No conflicts with system Python or other projects
- Multi-service: docker-compose runs API + Frontend together
- Portability: Push to Docker Hub, pull anywhere
- Industry standard: Most companies deploy with containers

**Key optimization**: Used CPU-only PyTorch in Docker (200MB vs 2.5GB GPU version), saving 2.3GB in image size.

**Tradeoff**: Docker adds complexity (Dockerfile, .dockerignore, compose) and the image is large (4.2GB with all data). But the reproducibility benefit is worth it.

---

## 8. Why Balanced Batch Sampling (not random sampling)?

**Choice**: P=16 classes x K=4 images = 64 per batch

**Alternatives considered**:
- Random sampling: Some batches might have only 1-2 classes → no valid triplets
- Hard example mining across full dataset: Too expensive (O(n²))
- Online pair mining: Less information per batch than triplets

**Reasoning**:
- Guarantees at least 4 images per class in every batch
- Batch hard mining needs multiple images per class to find hard negatives
- P=16 gives diversity (16 different categories per batch)
- K=4 gives enough positives for meaningful triplet selection
- Total batch size 64 fits comfortably in GPU memory

**Tradeoff**: Balanced sampling means we don't see rare classes as often (some categories have 7000+ images, others have <100). We mitigated this with stratified train/val/test splits.

---

## 9. Why L2 Distance → Cosine Similarity Conversion?

**Choice**: Store L2 distances in FAISS, convert to cosine similarity for display

**Reasoning**:
- FAISS is optimized for L2 distance (fastest implementation)
- Our embeddings are L2-normalized (magnitude = 1.0)
- For normalized vectors: L2² = 2(1 - cosine_similarity)
- So: cosine_sim = 1 - L2² / 2
- Users understand "99.7% similar" better than "L2 distance: 0.0548"

**Mathematical proof**:
```
Given: ||a|| = ||b|| = 1 (L2 normalized)
L2² = ||a - b||² = ||a||² + ||b||² - 2(a·b) = 2 - 2cos(θ)
Therefore: cos(θ) = 1 - L2²/2
```

---

## 10. Why Early Stopping with Patience=3?

**Choice**: Stop training if validation loss doesn't improve for 3 consecutive epochs

**Alternatives considered**:
- Fixed epochs: Risk of overfitting or underfitting
- Patience=1: Too aggressive, might stop during normal fluctuations
- Patience=5: Too lenient, wastes compute on diminishing returns

**Reasoning**:
- Validation loss naturally fluctuates between epochs
- Patience=3 allows temporary increases without premature stopping
- Saves model checkpoint at best validation loss
- Loads best weights after stopping (not the last epoch's weights)
- Result: Trained 11 epochs, stopped at 8, best was epoch 5

**Tradeoff**: With patience=3, we might miss a recovery after a longer plateau. But for our dataset size and model, 3 epochs of no improvement strongly indicates convergence.
