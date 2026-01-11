# Visual Guide: Architecture & Data Flow

## 1. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MULTIMODAL MEDICAL ANALYSIS SYSTEM                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ENCODER PATH (SigLIP - Image Understanding)                       │
│  ════════════════════════════════════════════════════════════     │
│                                                                     │
│     Medical Image (JPG/PNG)                                         │
│            ↓                                                         │
│     ImagePreprocessor                                              │
│     • Resize to 448×448                                             │
│     • Normalize (ImageNet)                                          │
│            ↓                                                         │
│     Vision Transformer (ViT)                                        │
│     • 28×28 patch grid (16×16 patches)                              │
│     • 12 layers, 768D embeddings                                    │
│            ↓                                                         │
│     Image Embedding [768D]                                          │
│                                                                     │
│  ══════════════════════════════════════════════════════════════   │
│                                                                     │
│  TEXT ENCODER PATH (SigLIP - Text Understanding)                   │
│  ════════════════════════════════════════════════════════════     │
│                                                                     │
│     Medical Description (Text)                                      │
│            ↓                                                         │
│     TextPreprocessor                                               │
│     • Tokenize (max 64 tokens)                                      │
│     • Create attention mask                                         │
│            ↓                                                         │
│     Transformer Text Encoder                                        │
│     • 12 layers, 768D embeddings                                    │
│            ↓                                                         │
│     Text Embedding [768D]                                           │
│                                                                     │
│  ══════════════════════════════════════════════════════════════   │
│                                                                     │
│  ALIGNMENT LAYER (Sigmoid Contrastive Loss)                        │
│  ════════════════════════════════════════════════════════════     │
│                                                                     │
│     Image Embedding ←→ Text Embedding                              │
│            ↓                                                         │
│     Cosine Similarity Matrix                                        │
│            ↓                                                         │
│     Sigmoid Contrastive Loss                                        │
│     • Learnable temperature                                         │
│     • Learnable bias                                                │
│            ↓                                                         │
│     Gradient Flow → Both Encoders                                   │
│                                                                     │
│  ══════════════════════════════════════════════════════════════   │
│                                                                     │
│  DECODER PATH (Llama 3.2 Vision - Generation)                      │
│  ════════════════════════════════════════════════════════════     │
│                                                                     │
│     Medical Image (JPG/PNG)                                         │
│            ↓                                                         │
│     MllamaProcessor                                                │
│     • Resize to 560×560                                             │
│     • Split into 4 tiles                                            │
│     • Generate aspect_ratio_ids/mask                                │
│            ↓                                                         │
│     Vision Encoder (32 layers)                                      │
│     • Process tiled image features                                  │
│            ↓                                                         │
│     Vision Features [B, 1, 4, 1024]                                │
│                                                                     │
│     Medical Description (Text)                                      │
│            ↓                                                         │
│     Tokenizer                                                       │
│     • Tokenize (max 512 tokens)                                     │
│     • Create labels for training                                    │
│            ↓                                                         │
│     Text Embedding Layer                                            │
│     • 128,256 vocab size                                            │
│     • 4096D embeddings                                              │
│            ↓                                                         │
│     Transformer Decoder (80 layers)                                 │
│     • Self-attention (causal masking)                               │
│     • Cross-attention to vision                                     │
│     • Feed-forward networks                                         │
│            ↓                                                         │
│     LM Head                                                          │
│            ↓                                                         │
│     Output Logits [B, seq_len, vocab_size]                          │
│            ↓                                                         │
│     Cross-Entropy Loss (ignores padding)                            │
│            ↓                                                         │
│     Gradient Flow → All Layers                                      │
│                                                                     │
│  ══════════════════════════════════════════════════════════════   │
│                                                                     │
│  INFERENCE PIPELINE                                                │
│  ════════════════════════════════════════════════════════════     │
│                                                                     │
│     New Medical Image                                               │
│            ↓                                                         │
│     Vision Encoder (SigLIP)                                         │
│            ↓                                                         │
│     Dense Image Embedding                                           │
│            ↓                                                         │
│     Decoder + Vision Encoder (Llama)                                │
│            ↓                                                         │
│     Medical Report/Description                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. SigLIP Contrastive Learning Visualization

```
Batch of 4 Image-Text Pairs:

Image 1: Knee X-ray (Normal)     ← matches → Text 1: "Normal joint"
Image 2: Skin lesion (melanoma)  ← matches → Text 2: "Melanoma"
Image 3: Chest X-ray (normal)    ← matches → Text 3: "Clear lungs"
Image 4: CT scan (abnormal)      ← matches → Text 4: "Abnormal finding"

                     ↓

Encode to 768D embeddings:

Vision Embeddings [4, 768]         Text Embeddings [4, 768]
    V1 [768D]                           T1 [768D]
    V2 [768D]                           T2 [768D]
    V3 [768D]                           T3 [768D]
    V4 [768D]                           T4 [768D]

                     ↓

Compute Similarity Matrix:

         T1      T2      T3      T4
    ┌────────────────────────────────┐
V1  │ 0.92    0.15    0.20    0.18   │  ← high on diagonal (positive)
V2  │ 0.18    0.89    0.22    0.16   │
V3  │ 0.16    0.19    0.91    0.14   │
V4  │ 0.14    0.17    0.15    0.88   │  ← high on diagonal
    └────────────────────────────────┘
         ↑
    diagonal = matched pairs (should be high)
    off-diagonal = mismatched (should be low)

                     ↓

Loss Computation (Sigmoid Contrastive):

For each row:
  Loss_i = -log(sigmoid(S_ii - log(sum_j exp(S_ij))))

Loss_1 = -log(sigmoid(0.92 - log(0.92+0.15+0.20+0.18)))
       = -log(sigmoid(0.92 - (-0.14)))
       = small value (good match!)

                     ↓

Backpropagate gradients:
- Increase diagonal elements (positive pairs)
- Decrease off-diagonal elements (negative pairs)
```

## 3. Llama 3.2 Vision Data Flow

```
Input Processing:
═════════════════

Medical Image (JPG, various sizes)
    ↓
┌──────────────────────────────────────────────┐
│ MllamaProcessor                              │
├──────────────────────────────────────────────┤
│ 1. Resize to 560×560                         │
│ 2. Normalize: (x - mean) / std               │
│ 3. Create aspect ratio ID (0-3)              │
│ 4. Split into tiles for larger images        │
│ 5. Generate aspect ratio mask                │
└──────────────────────────────────────────────┘
    ↓
Output: pixel_values [1, 1, 4, 3, 560, 560]
        aspect_ratio_ids [1, 1]
        aspect_ratio_mask [1, 1, 4]


Text Processing:
════════════════

Medical Description Text
    ↓
┌──────────────────────────────────────────────┐
│ Tokenizer                                    │
├──────────────────────────────────────────────┤
│ 1. Tokenize text                             │
│ 2. Truncate/Pad to 512 tokens                │
│ 3. Create attention mask                     │
│ 4. Create labels (input_ids → labels)        │
│ 5. Set padding positions to -100 in labels   │
└──────────────────────────────────────────────┘
    ↓
Output: input_ids [1, seq_len]
        attention_mask [1, seq_len]
        labels [1, seq_len]  (for training)


Model Forward Pass:
═══════════════════

Vision Path:
                pixel_values [1,1,4,3,560,560]
                     ↓
        ┌─────────────────────────────────┐
        │ Patch Embedding + Positional    │
        ├─────────────────────────────────┤
        │ [1, 1, 4, num_patches, 1024]    │
        └─────────────────────────────────┘
                     ↓
        ┌─────────────────────────────────┐
        │ Vision Transformer (32 layers)  │
        ├─────────────────────────────────┤
        │ • Self-attention                │
        │ • Feed-forward                  │
        │ • Normalization                 │
        └─────────────────────────────────┘
                     ↓
        Vision Output [1, 1, 4, 1024]


Text Path:
            input_ids [1, seq_len]
                 ↓
        ┌─────────────────────────────────┐
        │ Token Embedding                 │
        ├─────────────────────────────────┤
        │ [1, seq_len, 4096]              │
        └─────────────────────────────────┘
                 ↓
        ┌─────────────────────────────────┐
        │ Position Embedding (RoPE)       │
        ├─────────────────────────────────┤
        │ [1, seq_len, 4096]              │
        └─────────────────────────────────┘
                 ↓
            ↓ ↓ ↓ (80 Decoder Layers) ↓ ↓ ↓


For Each Decoder Layer:
═══════════════════════

    Hidden State [1, seq_len, 4096]
              ↙          ↓          ↘
          Self-Attn   Norm    Cross-Attn
          (text)      |      (image & text)
            ↓         ↓            ↓
    ┌─────────────────────────────────┐
    │ Self-Attention                  │
    │ • Query: from text hidden       │
    │ • Key/Value: from text hidden   │
    │ • Causal mask (can't see future)│
    │ Output: [1, seq_len, 4096]      │
    └─────────────────────────────────┘
              ↓
    ┌─────────────────────────────────┐
    │ Cross-Attention                 │
    │ • Query: from text              │
    │ • Key/Value: from vision        │
    │ • Allows text to "look at"      │
    │   image features                │
    │ Output: [1, seq_len, 4096]      │
    └─────────────────────────────────┘
              ↓
    ┌─────────────────────────────────┐
    │ Feed-Forward Network            │
    │ • 2 linear layers               │
    │ • GELU activation               │
    │ Output: [1, seq_len, 4096]      │
    └─────────────────────────────────┘
              ↓
        (repeat 80x)


Final Output:
═════════════

        Hidden State [1, seq_len, 4096]
              ↓
        ┌─────────────────────────────────┐
        │ Layer Norm                      │
        ├─────────────────────────────────┤
        │ [1, seq_len, 4096]              │
        └─────────────────────────────────┘
              ↓
        ┌─────────────────────────────────┐
        │ LM Head (Linear Projection)      │
        ├─────────────────────────────────┤
        │ [1, seq_len, vocab_size]        │
        │ vocab_size = 128,256            │
        └─────────────────────────────────┘
              ↓
        ┌─────────────────────────────────┐
        │ Loss Computation                │
        ├─────────────────────────────────┤
        │ Cross-Entropy(logits, labels)   │
        │ Ignore padding (label == -100)  │
        │ Average over real tokens        │
        └─────────────────────────────────┘
              ↓
        Scalar Loss (e.g., 11.3 → 5.9)
```

## 4. Training Loop Iteration

```
┌────────────────────────────────────────────────────────────┐
│ Training Loop for Fine-tuning Decoder                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  For epoch in range(num_epochs):                           │
│                                                            │
│    ┌──────────────────────────────────────────────────┐   │
│    │ For step, batch in dataloader:                   │   │
│    │                                                  │   │
│    │  Step 0: Load 1 image-text pair                  │   │
│    │  ├─ Image: Medical X-ray [1, 3, H, W]           │   │
│    │  ├─ Text: Description [1, seq_len]              │   │
│    │  └─ Process through MllamaProcessor              │   │
│    │     → pixel_values [1, 1, 4, 3, 560, 560]       │   │
│    │     → input_ids [1, seq_len]                     │   │
│    │     → labels [1, seq_len]                        │   │
│    │                                                  │   │
│    │  Forward Pass:                                   │   │
│    │  ├─ Encode image & text                          │   │
│    │  ├─ Compute logits for all positions             │   │
│    │  ├─ Compute cross-entropy loss                   │   │
│    │  └─ Divide by gradient_accum_steps (4)           │   │
│    │                                                  │   │
│    │  Backward Pass:                                  │   │
│    │  ├─ loss.backward()                              │   │
│    │  └─ Accumulate gradients                         │   │
│    │                                                  │   │
│    │  Step 3: Repeat (3 more batches to accumulate)   │   │
│    │                                                  │   │
│    │  After 4 accumulated batches:                    │   │
│    │  ├─ Clip gradients (max_norm=1.0)                │   │
│    │  ├─ Optimizer step                               │   │
│    │  ├─ Learning rate scheduler step                 │   │
│    │  ├─ Zero gradients                               │   │
│    │  └─ Log: "Step 1: Loss = 11.3"                   │   │
│    │                                                  │   │
│    │  Step 4-7: Repeat above...                       │   │
│    │                                                  │   │
│    │  After Step 8 (and every 4 steps):               │   │
│    │  ├─ Update step counter                          │   │
│    │  └─ If loss <= best_loss: Save checkpoint        │   │
│    │                                                  │   │
│    └──────────────────────────────────────────────────┘   │
│                                                            │
│    End of epoch: Print "Epoch 1/1 completed"              │
│                                                            │
│  Save Final Model:                                        │
│  └─ checkpoints/decoder_finetuned/pytorch_model.bin       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 5. Gradient Accumulation Visualization

```
Physical Batches: 4 samples × 1 step = 1 sample at a time

┌─────────────────────────────────────────────────────┐
│ STEP 1 (Batch element 1)                            │
├─────────────────────────────────────────────────────┤
│ Forward:  img₁ → logits₁ → loss₁ / 4 = 2.8         │
│ Backward: loss₁.backward()                          │
│ Gradients accumulated: ∇₁                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ STEP 2 (Batch element 2)                            │
├─────────────────────────────────────────────────────┤
│ Forward:  img₂ → logits₂ → loss₂ / 4 = 3.1         │
│ Backward: loss₂.backward()                          │
│ Gradients accumulated: ∇₁ + ∇₂                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ STEP 3 (Batch element 3)                            │
├─────────────────────────────────────────────────────┤
│ Forward:  img₃ → logits₃ → loss₃ / 4 = 2.5         │
│ Backward: loss₃.backward()                          │
│ Gradients accumulated: ∇₁ + ∇₂ + ∇₃                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ STEP 4 (Batch element 4)                            │
├─────────────────────────────────────────────────────┤
│ Forward:  img₄ → logits₄ → loss₄ / 4 = 2.9         │
│ Backward: loss₄.backward()                          │
│ Gradients accumulated: ∇₁ + ∇₂ + ∇₃ + ∇₄           │
└─────────────────────────────────────────────────────┘

                        ↓

        ┌──────────────────────────────────┐
        │ GRADIENT UPDATE (After Step 4)   │
        ├──────────────────────────────────┤
        │ ∇_total = (∇₁ + ∇₂ + ∇₃ + ∇₄)   │
        │ θ = θ - learning_rate × ∇_total │
        │                                  │
        │ Equivalent to processing         │
        │ effective batch of 4 at once!    │
        └──────────────────────────────────┘

        ↓

┌─────────────────────────────────────────────────────┐
│ STEP 5 (Reset, start new cycle)                     │
├─────────────────────────────────────────────────────┤
│ optimizer.zero_grad()                               │
│ Gradients cleared for next iteration                │
└─────────────────────────────────────────────────────┘
```

## 6. Loss Convergence Curve (Predicted)

```
Loss over training steps

|
| 12 ┌─ Batch 1: 11.30 (random prediction)
| 11 │
| 10 │  ─ Loss decreasing
|  9 │     ╲
|  8 │      ╲ Batch 2: 5.91
|  7 │       ╲
|  6 │        ╲
|  5 │         ╲     ─ Steady decrease
|  4 │          ╲   ╱
|  3 │           ╲╱  ─ Convergence region
|  2 │            ╲
|  1 │             └─ Target loss
|  0 └──────────────────────────
     0    100   200   300   400   500  steps
```

Expected trajectory:
- Steps 0-100: Rapid decrease (11.3 → 8.0)
- Steps 100-300: Medium decrease (8.0 → 4.5)
- Steps 300-500: Slow decrease (4.5 → 2.5)
- Steps 500+: Plateau (2.5 ± 0.5)
