# Technical Deep-Dive: Mathematical Foundations

## 1. SigLIP Loss Function

### 1.1 Sigmoid Contrastive Loss Derivation

#### Problem Statement
We want to learn embeddings $e_v \in \mathbb{R}^d$ (vision) and $e_t \in \mathbb{R}^d$ (text) such that:
- Matched pairs $(i, i)$ have high similarity
- Unmatched pairs $(i, j), i \neq j$ have low similarity

#### Mathematical Formulation
Given a batch of $N$ image-text pairs:

**Similarity Matrix:**
$$S_{ij} = \text{sim}(e_v^i, e_t^j) = \frac{e_v^i \cdot e_t^j}{\|e_v^i\| \|e_t^j\|}$$

**Sigmoid Contrastive Loss (per sample):**
$$\mathcal{L}_i = -\log \sigma(S_{ii} - \log \sum_j e^{S_{ij}})$$

Where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $S_{ii}$ is the positive pair similarity (diagonal)
- $\sum_j e^{S_{ij}}$ is the normalization factor

**Batch Loss:**
$$\mathcal{L} = \frac{1}{N} \sum_i \mathcal{L}_i$$

#### Why Sigmoid Instead of Softmax?

**Softmax (CLIP):**
- Fixed temperature parameter
- Requires balancing matching with negative mining
- All negatives equally important

**Sigmoid (SigLIP):**
- Learnable temperature parameter $\tau$
- Only positive pair matters
- More stable for medical domain with varied concepts

#### With Learnable Parameters

**Modified Loss:**
$$\mathcal{L} = -\frac{1}{N} \sum_i \log \sigma(\tau(S_{ii} - \log \sum_j e^{S_{ij}}) + b)$$

Where:
- $\tau = e^{t}$ (temperature, $t$ is learnable)
- $b$ is learnable bias

**Typical values for medical domain:**
- Initial $\tau = \log(10) \approx 2.3$
- Initial $b = -10$

### 1.2 Gradient Flow in SigLIP

**Vision Encoder Gradient:**
$$\frac{\partial \mathcal{L}}{\partial e_v^i} = \frac{\partial \mathcal{L}}{\partial S_{ii}} \cdot \frac{\partial S_{ii}}{\partial e_v^i} + \sum_j \frac{\partial \mathcal{L}}{\partial S_{ij}} \cdot \frac{\partial S_{ij}}{\partial e_v^i}$$

**Text Encoder Gradient:**
$$\frac{\partial \mathcal{L}}{\partial e_t^j} = \frac{\partial \mathcal{L}}{\partial S_{jj}} \cdot \frac{\partial S_{jj}}{\partial e_t^j} + \sum_i \frac{\partial \mathcal{L}}{\partial S_{ij}} \cdot \frac{\partial S_{ij}}{\partial e_t^j}$$

Both encoders receive gradients to:
1. Increase positive similarity $S_{ii}$
2. Decrease negative similarities $S_{ij}, i \neq j$

---

## 2. Llama 3.2 Vision Cross-Entropy Loss

### 2.1 Causal Language Modeling

**Objective:** Given image $\mathbf{x}$ and context tokens $t_1, \ldots, t_{n-1}$, predict $t_n$

**Sequence:**
$$P(t_n | \mathbf{x}, t_1, \ldots, t_{n-1}) = \text{softmax}(\mathbf{W} h_n + b)_{\text{token}}$$

Where:
- $h_n$ is the hidden state at position $n$
- $\mathbf{W}$ is the output projection matrix
- $h_n$ depends on vision features and previous tokens (via causal attention)

### 2.2 Cross-Entropy Loss

**Per-token loss:**
$$\ell_n = -\log P(t_n | \mathbf{x}, t_1, \ldots, t_{n-1})$$

**Batch loss:**
$$\mathcal{L} = \frac{1}{T_{real}} \sum_{n=1}^{T} \ell_n \cdot \mathbb{1}_{t_n \neq \text{pad}}$$

Where:
- $T$ is sequence length
- $T_{real}$ is number of non-padding tokens
- $\mathbb{1}_{t_n \neq \text{pad}}$ is indicator (ignores padding)

### 2.3 Attention Mechanism with Image Features

**Self-attention on text:**
$$\text{Attn}_{\text{self}}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Cross-attention to vision:**
$$\text{Attn}_{\text{cross}}(Q_{\text{text}}, K_{\text{vision}}, V_{\text{vision}}) = \text{softmax}\left(\frac{Q_{\text{text}}K_{\text{vision}}^T}{\sqrt{d_k}}\right)V_{\text{vision}}$$

**Combined in each decoder layer:**
$$h'_n = \text{Attn}_{\text{self}}(h_n) + \text{Attn}_{\text{cross}}(h_n, \text{vision\_features})$$

---

## 3. Vision Transformer Patch Embedding

### 3.1 Patch Division (for Llama 3.2)

**Input image:** $\mathbf{x} \in \mathbb{R}^{3 \times H \times W}$

Example: medical X-ray $560 \times 560$ pixels

**Patch division:**
- Patch size: $p = 140$ (creating 4×4 grid)
- Number of patches: $N_p = (H/p)^2 = 4 \times 4 = 16$ patches

But Llama 3.2 uses **tiling** approach:
- Resize to $560 \times 560$
- Split into **4 tiles** (not 16)
- Each tile: $280 \times 280$ (approximate)

### 3.2 Patch Embedding

**For each patch:**
$$\mathbf{p}_{i,j} = \mathbf{x}[:, i \cdot p:(i+1) \cdot p, j \cdot p:(j+1) \cdot p]$$

**Linear embedding:**
$$e_{i,j} = \mathbf{W} \text{flatten}(\mathbf{p}_{i,j}) + b$$

Where:
- $\mathbf{W} \in \mathbb{R}^{d \times (3 \cdot p^2)}$
- Output: $e_{i,j} \in \mathbb{R}^d$

**Vision transformer processes:**
$$[\text{CLS}] + [e_{1,1}, e_{1,2}, \ldots, e_{n,n}] + \text{pos\_encoding}$$

---

## 4. Gradient Accumulation Mathematics

### 4.1 Problem: Large Batch Size on Limited Memory

**Without accumulation:** Need batch size $B$ simultaneously in memory

**Solution:** Process batch in $k$ smaller steps

### 4.2 Mathematical Equivalence

**With batch size $B$ and $k$ steps:**

$$\mathcal{L}_{\text{total}} = \frac{1}{k} \sum_{i=1}^{k} \mathcal{L}_i \quad \text{(averaged)}$$

Where each $\mathcal{L}_i$ is computed on $B/k$ samples

**Gradient update (equivalent):**

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L}_{\text{total}}$$

**Implementation:**
```python
# Step 1: Forward & backward on subset 1
loss_1 = model(batch_1) / k
loss_1.backward()  # gradients accumulate

# Step 2: Forward & backward on subset 2
loss_2 = model(batch_2) / k
loss_2.backward()  # add to accumulated gradients

# Step k: Optimizer step on accumulated gradients
optimizer.step()
optimizer.zero_grad()
```

### 4.3 Effective Batch Size

Our configuration:
- Physical batch: 1 sample
- Accumulation steps: 4
- Effective batch: 4 samples

$$\text{Effective Batch} = \text{Physical Batch} \times \text{Accumulation Steps}$$

$$= 1 \times 4 = 4$$

---

## 5. Learning Rate Scheduling

### 5.1 Cosine Annealing with Warmup

**Warmup phase** ($t < t_{\text{warmup}}$):
$$\eta_t = \eta_{\text{max}} \cdot \frac{t}{t_{\text{warmup}}}$$

**Cosine phase** ($t \geq t_{\text{warmup}}$):
$$\eta_t = \frac{\eta_{\text{min}} + \eta_{\text{max}}}{2} + \frac{\eta_{\text{max}} - \eta_{\text{min}}}{2} \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T_{\text{total}} - t_{\text{warmup}}}\right)$$

### 5.2 For SigLIP Training

**Configuration:**
- $\eta_{\text{max}} = 1 \times 10^{-4}$
- $t_{\text{warmup}} = 0.1 \times T_{\text{total}}$
- $T_{\text{total}} = \text{num\_batches}$

**Example with 600 optimizer steps:**
- Warmup: steps 0-60 (gradually increase LR)
- Cosine decay: steps 60-600 (exponentially decay)

---

## 6. Mixed Precision Training

### 6.1 Why Mixed Precision?

**Full precision (fp32):**
- Accurate but slow
- Uses more memory

**Half precision (fp16):**
- Faster computation
- Less memory
- Risk of numerical underflow/overflow

**Solution:** Mixed precision
- Forward pass: fp16 (fast)
- Loss scaling: multiply by large factor
- Backward pass: compute gradients in fp32
- Update: apply to fp32 parameters

### 6.2 Implementation with GradScaler

```
Forward Pass (fp16):
logits = model(input)  # fp16 operations, fast
loss = criterion(logits, target)  # fp16

Loss Scaling:
scaled_loss = loss * scale_factor  # prevent underflow

Backward Pass (fp32):
scaler.scale(scaled_loss).backward()  # compute gradients

Unscaling & Step:
scaler.unscale_(optimizer)  # remove scale factor
clip_grad_norm_(model.parameters())  # gradient clipping
scaler.step(optimizer)  # apply update
scaler.update()  # update scale factor for next iteration
```

### 6.3 Benefits

| Aspect | Benefit |
|--------|---------|
| Memory | ~50% reduction |
| Speed | ~2× faster |
| Accuracy | Minimal loss (>99.9% same) |

---

## 7. Aspect Ratio Handling in Llama 3.2

### 7.1 Problem: Variable Image Sizes

Different medical images have different aspect ratios:
- X-rays: often square (512×512)
- Skin lesion: various (rectangular)
- Anatomical: various sizes

### 7.2 Solution: Aspect Ratio IDs

**Categorization:**
```python
aspect_ratio_categories = {
    'square': [0.8, 1.2],      # 0.8 < AR < 1.2
    'landscape': [1.2, 2.0],   # wide images
    'portrait': [0.5, 0.8],    # tall images
    'extreme': [2.0, 5.0]      # very wide/tall
}
```

**For each image, compute:**
$$\text{aspect\_ratio} = \frac{\text{width}}{\text{height}}$$

**Assign ID:** 0, 1, 2, or 3

**Benefit:** Model learns aspect-ratio-specific attention patterns

### 7.3 Aspect Ratio Mask

**Purpose:** Track valid tiles in batch

For batch with variable-sized images:
```
Image 1: 500×500 (square) → 4 valid tiles
Image 2: 600×300 (landscape) → 4 valid tiles
Image 3: 400×500 (portrait) → 4 valid tiles

aspect_ratio_mask shape: [batch_size, 1, 4]
                           ↓           ↓   ↓
                        batch  num_images  tiles

If image is too small, some tile positions padded with zeros
```

---

## 8. Tensor Shapes Through the Pipeline

### 8.1 SigLIP Training

```
Input: PIL Image (arbitrary size) + Text
  ↓
Processor:
  - Resize to 448×448
  - Normalize: (img - mean) / std
  - Tokenize text: max 64 tokens
  ↓
Tensors:
  - pixel_values: [B, 3, 448, 448]
  - input_ids: [B, 64]
  - attention_mask: [B, 64]
  ↓
Model:
  - Vision Encoder: [B, 3, 448, 448] → [B, 768]
  - Text Encoder: [B, 64] → [B, 768]
  - Cosine similarity: [B, B]
  ↓
Loss: scalar
  - Sigmoid contrastive loss
```

### 8.2 Llama 3.2 Vision Training

```
Input: PIL Image (arbitrary size) + Text
  ↓
Processor:
  - Resize to 560×560
  - Split into 4×4=4 tiles (tiling strategy)
  - Normalize with ImageNet stats
  - Tokenize text: max 512 tokens
  ↓
Tensors:
  - pixel_values: [B, 1, 4, 3, 560, 560]
    ├─ B: batch size
    ├─ 1: one image per batch element
    ├─ 4: number of tiles
    ├─ 3: RGB channels
    └─ 560×560: tile resolution
  
  - input_ids: [B, seq_len]
  - attention_mask: [B, seq_len]
  - aspect_ratio_ids: [B, 1]
  - aspect_ratio_mask: [B, 1, 4]
  - labels: [B, seq_len]
  ↓
Model:
  - Vision Encoder: [B, 1, 4, 3, 560, 560] → features
  - Text Embedding: [B, seq_len] → [B, seq_len, 4096]
  - Transformer Decoder: with cross-attention
  - LM Head: → [B, seq_len, vocab_size]
  ↓
Loss: scalar
  - Cross-entropy loss (only on real tokens)
```

---

## 9. Perplexity and Loss Interpretation

### 9.1 Perplexity Definition

For language models:
$$\text{Perplexity} = e^{\text{average cross-entropy loss}}$$

$$= e^{\frac{1}{T} \sum_t -\log P(t_i | \text{context})}$$

### 9.2 Interpretation

| Loss | Perplexity | Meaning |
|------|-----------|---------|
| 11.5 | 100,000 | Completely random, 1 in 100k chance |
| 6.0 | 403 | Very uncertain, 1 in 403 average |
| 3.0 | 20 | Decent prediction, 1 in 20 chance |
| 2.0 | 7.4 | Good prediction, 1 in 7 chance |
| 1.0 | 2.7 | Very good, narrow distribution |

### 9.3 Our Decoder Results

```
Batch 1: Loss = 11.30 → Perplexity = 80,000
Batch 2: Loss = 5.91 → Perplexity = 370

Improvement factor: 80,000 / 370 ≈ 216×
The model is learning rapidly!
```

---

## 10. Optimization Dynamics

### 10.1 AdamW Optimizer

**Update rule:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L})^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}$$

**Parameters in our training:**
- $\beta_1 = 0.9$ (first moment decay)
- $\beta_2 = 0.95$ (second moment decay) - reduces to 0.999 from default
- $\epsilon = 1e-8$ (numerical stability)
- $\lambda = 0.01$ (weight decay)

### 10.2 Gradient Clipping

**Prevents exploding gradients:**
$$\nabla \mathcal{L}_{\text{clipped}} = \nabla \mathcal{L} \cdot \min(1, \frac{\text{max\_norm}}{||\nabla \mathcal{L}||})$$

**Our setting:** `max_grad_norm = 1.0`

---

## Summary of Key Equations

| Component | Equation |
|-----------|----------|
| SigLIP Loss | $\mathcal{L} = -\frac{1}{N}\sum_i \log \sigma(\tau(S_{ii} - \log\sum_j e^{S_{ij}}) + b)$ |
| Cross-Entropy | $\mathcal{L} = -\frac{1}{T_r}\sum_n \mathbb{1}_{t_n \neq \text{pad}} \log P(t_n \| x, t_{<n})$ |
| Gradient Update | $\theta_t = \theta_{t-1} - \eta \nabla \mathcal{L}$ |
| Effective Batch | $B_{\text{eff}} = B_{\text{phys}} \times \text{accum\_steps}$ |
| Cosine LR | $\eta_t = \eta_{\max} \cos(\pi \cdot \frac{t-t_w}{T-t_w})$ |
| Perplexity | $\text{PPL} = e^{\mathcal{L}}$ |
