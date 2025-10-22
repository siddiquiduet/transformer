# Layer Normalization in Transformers: A Concise Tutorial
Source: https://youtu.be/W9g7j22MO-Q?si=skHsewXJ60kNL3I6
## Why Normalization Matters

Training deep Transformers with billions of parameters faces critical challenges: vanishing/exploding gradients and unstable optimization. **Layer Normalization** solves these problems.

### The Unnormalized Data Problem

Consider predicting house prices using:
- Square footage: 500-5,000
- Number of rooms: 1-10

This scale disparity creates an **elongated cost function**—like a narrow canyon versus a smooth bowl.

<img width="259" height="150" alt="elongated_cost_fun" src="https://github.com/user-attachments/assets/745e7908-9868-44b6-8d58-66f51fa6c7a8" />


**Consequences:**
- High learning rate → Overshooting
- Low learning rate → Slow convergence  
- High variance → Gradient instability

**Solution:** Normalization transforms data to have mean = 0, std = 1

```
Formula: z_norm = (z - μ) / σ

Example:
Original: [1000, 1500, 2000, 2500, 3000]
μ = 2000, σ = 707.1
Normalized: [-1.41, -0.71, 0.00, 0.71, 1.41]
```

Result: Circular cost function → faster, stable training.

<img width="150" height="150" alt="circular" src="https://github.com/user-attachments/assets/0f60078b-c04f-45c7-bd92-938e6e96745a" />


## Internal Covariate Shift: The Deep Network Challenge
<img width="515" height="84" alt="image" src="https://github.com/user-attachments/assets/32ab0c4a-3018-489b-aaec-05014c02ac0a" />

Normalizing only input isn't enough. In deep networks:
1. Weights update during backpropagation
2. Each layer's output distribution changes
3. Next layers receive a "moving target" every iteration

**Analogy:** Learning basketball where rules change daily—you never master fundamentals.

**Example:** Train on red roses → test on white roses = failure (covariate shift). In deep networks, this happens **internally at every layer, every iteration**.

## Layer Normalization: The Two-Stage Solution

### Stage 1: Normalize

```
For data point with features [z₁, z₂, ..., z_d]:

μ = (1/d) Σ(zⱼ)
σ = sqrt((1/d) Σ(zⱼ - μ)²)
z_norm = (z - μ) / (σ + ε)
```

### Stage 2: Scale and Shift

```
z_output = γ ⊙ z_norm + β

γ (gamma) = learnable scale (init: 1)
β (beta)  = learnable shift (init: 0)
```

**Why?** Flexibility! Not all layers need full normalization. The model **learns** optimal normalization through γ and β.

**Extreme case:** If γ = σ and β = μ → recovers original values (model chose no normalization).

## Batch Norm vs. Layer Norm

### Batch Normalization
Normalizes **across batch** for each feature:

```
3 samples, 5 features:
     F1   F2   F3   F4   F5
S1: [2.1, 3.4, 1.8, 4.2, 2.9]
S2: [1.9, 3.1, 2.1, 4.5, 3.2]
S3: [2.3, 3.6, 1.7, 4.1, 2.8]
     ↓    ↓    ↓    ↓    ↓
Calculate μ, σ for each column
```

### Layer Normalization
Normalizes **across features** for each sample:

```
S1: [2.1, 3.4, 1.8, 4.2, 2.9] → μ₁, σ₁
    ←─────────────────────────
S2: [1.9, 3.1, 2.1, 4.5, 3.2] → μ₂, σ₂
    ←─────────────────────────
Each row independent!
```

### The Padding Problem

**Why Transformers need Layer Norm:**

```
Batch: 2 sentences, max length 4, embedding 512
S1: "The cat sat" → [emb₁, emb₂, emb₃, PAD]
S2: "I love coding" → [emb₁, emb₂, emb₃, emb₄]

Batch Norm problem:
Token 4 normalization = mean([PAD_zeros, emb₄])
→ Zeros corrupt statistics! ✗

Layer Norm solution:
Each token normalized across its 512 features
→ No cross-contamination! ✓
```

## Application in Transformers

### Structure
**Input shape:** [batch_size, seq_length, embed_dim]  
Example: [32, 128, 768]

**For each token:**
```
Token: [e₁, e₂, ..., e₇₆₈]

μ = Σ(eᵢ) / 768
σ = sqrt(Σ(eᵢ - μ)² / 768)
e_norm = (e - μ) / σ
e_output = γ × e_norm + β

Note: γ, β shape = [768] (one per feature)
```

### Transformer Block
```
Input → Multi-Head Attention → Add & LayerNorm
     → Feed-Forward Network  → Add & LayerNorm
     → Output
```

**Modern models:** GPT-3 has 96 layers × 2 LayerNorms = **192 normalization operations per forward pass!**

## Concrete Example

```
Token output: [0.8, -1.2, 0.5, 2.1, -0.3]

Step 1: μ = 0.38, σ = 1.18

Step 2: Normalize
[0.8, -1.2, 0.5, 2.1, -0.3]
→ [0.36, -1.34, 0.10, 1.46, -0.58]

Step 3: Apply γ=[1.5,...], β=[0.1,...]
→ [0.64, -1.91, 0.25, 2.29, -0.77]
```

## Pre-Norm vs. Post-Norm

**Post-Norm (Original):** LayerNorm after sublayer  
**Pre-Norm (Modern):** LayerNorm before sublayer

```
Pre-Norm (GPT):
Input → LayerNorm → Attention → Add
     → LayerNorm → FFN → Add
```

**Why Pre-Norm?** More stable for 100+ layer networks, easier training.

## Key Takeaways

1. **Stabilizes training** by preventing gradient issues and enabling faster convergence
2. **Internal covariate shift** requires normalizing intermediate layers
3. **Layer Norm > Batch Norm** for Transformers (handles variable lengths/padding)
4. **Scale/shift (γ, β)** give models learned flexibility
5. **Essential for modern AI:** Every major Transformer relies on LayerNorm
6. **Without it:** Training 100+ layer Transformers would be impossible

## Practical Impact

**GPT-3:** 175B parameters, 96 layers, 192 LayerNorm ops/token  
**Training cost:** $4-12 million

Layer Normalization ensures gradients flow cleanly through this massive network. Without it, the investment would be wasted on unstable training.

---

**Bottom Line:** Layer Normalization transformed Transformers from theory to the foundation of modern AI. Simple concept, profound impact! 🚀
