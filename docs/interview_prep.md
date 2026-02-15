# Interview Q&A — Visual Search Engine

Common interview questions about this project and how to answer them.

---

## Section 1: System Design Questions

### Q1: "Walk me through the architecture of your visual search system."

**Answer:**
"The system has 4 main components:

1. **Embedding Model** — ResNet50 fine-tuned with triplet loss. Takes a 224x224 image, outputs a 2048-dimensional L2-normalized embedding vector.

2. **FAISS Index** — Stores 43,916 pre-computed product embeddings. Uses flat (exact) index with L2 distance. Search takes ~2ms.

3. **FastAPI Backend** — Three endpoints: health check, visual search (accepts image upload, returns top-K similar products), and image serving. Model and index load once at startup.

4. **Streamlit Frontend** — Upload interface that calls the API, displays results in a grid with similarity scores.

The flow: User uploads image → API extracts embedding → FAISS finds nearest neighbors → API enriches with metadata → Frontend displays results."

---

### Q2: "How would you scale this to 10 million products?"

**Answer:**
"Three key changes:

1. **Index**: Switch from FAISS Flat to IVF+PQ (Inverted File with Product Quantization). At 10M items, flat search is too slow. IVF partitions the space into clusters, PQ compresses vectors from 2048 floats (8KB each) to ~128 bytes. Index shrinks from ~80GB to ~1.2GB.

2. **Infrastructure**: Move from single server to distributed. Use FAISS's distributed search or a vector database like Milvus/Pinecone. Separate the embedding service (GPU) from the search service (CPU).

3. **Caching**: Add Redis cache for popular query embeddings. If someone searches for 'red dress' and we've seen similar embeddings before, return cached results instead of re-searching.

I benchmarked IVF and IVF+PQ during development. At 44K items, IVF maintained 100% recall at nprobe=10. PQ with m=64 dropped to 62% recall — not acceptable. At 10M, I would use IVF+OPQ (Optimized Product Quantization) which learns a rotation matrix to minimize quantization error."

---

### Q3: "Why did you choose ResNet50 over newer models like ViT?"

**Answer:**
"Three reasons:

1. **Inference speed**: ResNet50 runs in ~50ms on CPU. ViT-Base takes ~200ms. For real-time search, 4x speed difference matters.

2. **Embedding quality for retrieval**: ResNet50 with triplet loss achieves 82.6% P@5 on our dataset. ViT would likely be a few percentage points better, but the engineering complexity (larger model, longer training, more GPU memory) doesn't justify it at this scale.

3. **Deployment simplicity**: ResNet50 is 90MB. ViT-Base is 340MB. Smaller model = smaller Docker image = faster deployment.

If I were building this at a company with GPU inference servers and millions of products, I would benchmark ViT against ResNet50 and choose based on actual accuracy/latency tradeoffs."

---

### Q4: "What happens if the user uploads a non-fashion image (like a cat)?"

**Answer:**
"Currently, the system still returns results — it finds the 10 products whose embeddings are closest to the cat image's embedding. The scores would be low (maybe 30-40% similarity instead of 90%+).

To handle this properly, I would add:

1. **Score threshold**: If the top result's similarity score is below 0.5, return a message like 'No similar products found.'

2. **Category classifier**: Add a lightweight classifier head that predicts if the image is a fashion product. If confidence < 0.5, reject the upload.

3. **Out-of-distribution detection**: Compare the query embedding's distance to the nearest cluster centroid. If it's far from all clusters, flag it as out-of-distribution."

---

## Section 2: ML-Specific Questions

### Q5: "Explain triplet loss. Why is it better than cross-entropy for this task?"

**Answer:**
"Triplet loss works with three images at a time:
- **Anchor**: The reference image
- **Positive**: Same category as anchor
- **Negative**: Different category

The loss function is: max(0, d(anchor, positive) - d(anchor, negative) + margin)

It pushes the model to make same-category images CLOSER and different-category images FARTHER in embedding space.

Cross-entropy loss classifies images into categories. The problem is it doesn't directly optimize the embedding space. Two shirts classified as 'Shirts' might have embeddings far apart. Triplet loss explicitly forces them together.

Also, cross-entropy requires a fixed set of categories. If we add a new product category, we need to retrain. With triplet loss, new categories work immediately — just compute the embedding and add to the index."

---

### Q6: "What is batch hard mining?"

**Answer:**
"In a batch of 64 images (16 classes x 4 per class):

For each anchor image:
1. **Hardest positive**: The image from the SAME class that is FARTHEST away (hardest to recognize as similar)
2. **Hardest negative**: The image from a DIFFERENT class that is CLOSEST (hardest to distinguish)

These are the most informative examples — the model learns the most from mistakes it's about to make.

Without hard mining, most triplets are 'easy' — the model already gets them right. Easy triplets produce zero loss (the max(0, ...) clips to zero), so the model doesn't learn anything from them.

I tracked 'active triplets per batch' during training — it dropped from 53 to 19 over training, meaning the model was solving more triplets correctly as it learned."

---

### Q7: "Why L2 normalize embeddings?"

**Answer:**
"Three reasons:

1. **Fair comparison**: Without normalization, an image with larger activation magnitudes would dominate. Normalization ensures every image has equal 'weight' in the embedding space.

2. **L2 distance ↔ cosine similarity**: For normalized vectors, L2² = 2(1 - cos_sim). This means L2 nearest neighbors ARE cosine nearest neighbors. We get cosine similarity for free.

3. **Numerical stability**: Keeps values bounded. Without normalization, embeddings could have magnitudes of 0.01 or 1000, making distance thresholds meaningless."

---

### Q8: "Your Precision@5 is 82.6%. How would you improve it?"

**Answer:**
"Several approaches, ordered by expected impact:

1. **Unfreeze more layers** (expected: +3-5%): Currently we only train layer4. Unfreezing layer3 would allow the model to learn mid-level fashion features (textures, patterns).

2. **Multi-scale augmentation** (expected: +2-3%): Add random crops, color jitter, horizontal flips during training. Makes the model robust to variation.

3. **Larger model** (expected: +2-4%): Try ResNet101 or EfficientNet-B4. More parameters = more capacity to learn fine-grained differences.

4. **Better loss function** (expected: +1-3%): Try ArcFace or Multi-Similarity Loss. These have been shown to outperform vanilla triplet loss in recent retrieval benchmarks.

5. **Category-aware re-ranking** (expected: +2-3%): After FAISS returns top-50, re-rank using a lightweight model that considers both visual similarity and category compatibility."

---

## Section 3: Engineering Questions

### Q9: "How do you handle model updates in production?"

**Answer:**
"I would implement blue-green deployment:

1. Train new model → extract new embeddings → build new FAISS index
2. Deploy new model as 'green' service alongside existing 'blue' service
3. Route 10% of traffic to green (canary deployment)
4. Compare metrics: if green's P@5 >= blue's P@5, gradually increase traffic
5. Once green handles 100%, decommission blue

Key requirement: the new embeddings are INCOMPATIBLE with the old index. You must rebuild the entire index when the model changes. This is why embedding versioning is critical."

---

### Q10: "Why did you separate the API and frontend into different containers?"

**Answer:**
"Separation of concerns and independent scaling:

1. **Independent scaling**: If search traffic increases, I can spin up 5 API containers but keep 1 frontend. With a combined container, I'd waste resources running 5 frontends nobody uses.

2. **Independent deployment**: I can update the frontend (change layout, add features) without restarting the API. Users don't experience downtime.

3. **Different resource needs**: The API needs 2-4GB RAM (model + index). The frontend needs 256MB. Separating them allows right-sizing each container.

4. **Microservices pattern**: This is how most production systems are built. It demonstrates I understand modern deployment architecture."

---

### Q11: "How does your CI/CD pipeline work?"

**Answer:**
"Three-stage pipeline triggered on every push to main:

1. **Lint**: flake8 checks code quality — catches style issues, unused imports, syntax errors.

2. **Test**: pytest runs 47 tests covering configuration, embedding extraction, and all API endpoints. Tests verify correctness (right shapes, normalized embeddings, sorted results, proper error codes).

3. **Docker**: Verifies Dockerfile and docker-compose.yml exist, validates the image is available on Docker Hub.

The pipeline is sequential: lint must pass before test, test must pass before Docker. This catches errors early — no point testing broken code, no point building broken tests."

---

### Q12: "What would you add if you had more time?"

**Answer:**
"In priority order:

1. **Text search**: Add a text encoder (CLIP or Sentence-BERT) so users can search by description: 'red floral dress'. Combine text and image embeddings with learned weights.

2. **User feedback loop**: Track which results users click → use as implicit positive labels → periodic model retraining. This creates a flywheel: better model → better results → more clicks → better training data.

3. **A/B testing framework**: Compare different models, index configurations, and re-ranking strategies on live traffic with statistical significance.

4. **Monitoring dashboard**: Track P@5, latency percentiles (p50, p95, p99), error rates, and model drift over time. Alert if metrics degrade.

5. **Multi-modal search**: Combine visual features with metadata (color, brand, price range) for filtered search: 'shoes similar to this image, under $50, in black.'"

---

## Section 4: Quick-Fire Questions

### Q: "What's the time complexity of FAISS flat search?"
**A:** "O(n × d) where n=43,916 items and d=2048 dimensions. It's brute-force — compares query against every vector."

### Q: "What's the space complexity of your index?"
**A:** "O(n × d × 4) bytes = 43,916 × 2048 × 4 = ~343MB. Each float32 is 4 bytes."

### Q: "Why 2048 dimensions?"
**A:** "ResNet50's average pooling layer outputs 2048 features. We removed the classification head (fc layer) and use the pooling output directly as the embedding."

### Q: "What margin did you use for triplet loss? Why?"
**A:** "0.3. It means: the distance to the hardest negative must be at least 0.3 greater than the distance to the hardest positive. Too small (0.1) → embeddings are too close, poor discrimination. Too large (1.0) → too hard to satisfy, training is unstable. 0.3 is a commonly used default in retrieval literature."

### Q: "How many trainable parameters?"
**A:** "14.9M out of 25.5M total (58.6%). All in layer4 of ResNet50."

### Q: "What happens during a cold start?"
**A:** "Model loading takes ~30-40 seconds: download ResNet50 base weights (~98MB), load fine-tuned weights (~90MB), load FAISS index (~343MB), load metadata CSV. That's why we have a 60-second start_period in the Docker healthcheck."

### Q: "Why not use a vector database like Pinecone?"
**A:** "At 44K items, FAISS in-memory is simpler, faster, and free. Pinecone adds network latency (~10-50ms) and costs money. I would switch to a managed vector DB at 1M+ items where I need distributed search, automatic scaling, and backup."
