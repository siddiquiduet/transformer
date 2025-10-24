# Encoder Architecture in Transformers: A Comprehensive Tutorial
Source: https://youtu.be/DFqWPwF0OH0?si=1CPgoR5cwo01tJ3G

## Introduction

The Transformer encoder is a cornerstone of modern NLP architectures. Understanding its design reveals how models like BERT achieve remarkable performance in language understanding tasks. This tutorial breaks down the encoder's components systematically.

## 1. The Input Layer: Beyond Static Embeddings

### The Problem with Static Word Embeddings

Traditional word embeddings map words to fixed vectors, but language is inherently contextual. Consider these examples:

- "Turn on the **light**" (electromagnetic radiation)
- "This bag is **light**" (weight measurement)  
- "I prefer **light** blue" (color shade)

The same word "light" requires different representations based on context. Static embeddings fail to capture this nuance.

### The Solution: Dynamic Contextualization

The encoder transforms static embeddings into context-aware representations by analyzing relationships between all words in a sequence. When processing "I love Apple phones," the model shifts "Apple" toward technology semantics rather than fruit semantics.

## 2. Positional Encoding: Preserving Word Order

### Why Positional Information Matters

Since the encoder processes words in parallel (not sequentially), it loses inherent word order. Compare:

- "The cat chased the mouse"
- "The mouse chased the cat"

These sentences contain identical words but opposite meanings. Without positional information, they'd be indistinguishable.

### Implementation

Positional encodings are vectors generated using sinusoidal functions that create predictable patterns. These 512-dimensional vectors are **element-wise added** to word embeddings before entering the encoder.

**Formula**: The input to multi-head attention becomes:
```
Input = Word_Embedding + Positional_Encoding
```

The sinusoidal pattern allows the model to infer relative positions—if it knows position 10's encoding, it can extrapolate positions 20, 30, and beyond.

## 3. Multi-Head Self-Attention: The Core Mechanism

### Single Attention Head Limitations

One attention mechanism can capture one type of relationship pattern. For the sentence "Keys on the table belong to Sara," we need to capture:

1. **Spatial relationship**: "keys" ↔ "on the table"
2. **Possession relationship**: "keys" ↔ "belong to Sara"

A single attention head cannot effectively model both simultaneously.

### Multi-Head Architecture
<img width="872" height="496" alt="multi-headed-attention" src="https://github.com/user-attachments/assets/47b7d4ba-147f-41c9-94a2-413819f0caca" />

The original Transformer uses **8 parallel attention heads**:

- **Input**: Each head receives the same 512-dimensional embedding (3 × 512 for three words)
- **Processing**: Each head learns different relationship patterns
  - Head 1: Subject-verb-object relationships
  - Head 2: Spatial dependencies  
  - Head 3: Temporal patterns
  - Head 4-8: Other semantic relationships

- **Output**: Each head produces 64-dimensional representations (3 × 64)
- **Concatenation**: Eight 64-dimensional outputs concatenate to 512 dimensions (8 × 64 = 512)
- **Linear transformation**: A weight matrix W^o (512 × 512) projects concatenated outputs

### Parallel Processing Advantage

All words are processed simultaneously, enabling:
- GPU parallelization
- Constant processing time regardless of sequence length (10 words or 1,000 words take equal time)
- Dramatically faster training than sequential models (RNNs, LSTMs)

## 4. Add & Norm Layer: Stabilizing Deep Networks
<img width="791" height="450" alt="add_n_norm" src="https://github.com/user-attachments/assets/bbfeea5b-33a4-458a-a2ec-47cf9d55673d" />

### Residual Connections (The "Add" Part)
<img width="859" height="319" alt="residual_connection" src="https://github.com/user-attachments/assets/f267aabe-4db7-43ed-82f8-f1bd77ca4b26" />

Deep networks face a critical problem: as inputs pass through many layers with random initial weights, information becomes distorted into noise by the final layer.

**Solution**: Skip connections provide a shortcut path for information flow.

**Operation**:
```
Output = LayerNorm(Input + MultiHeadAttention(Input))
```

The input Z₀ is added element-wise to the multi-head attention output Z₀'. This requires matching dimensions (both 512-dimensional).

### Layer Normalization (The "Norm" Part)

Normalization stabilizes training by:
- Preventing vanishing gradients (gradients becoming too small)
- Preventing exploding gradients (gradients becoming too large)
- Ensuring faster convergence

Add & Norm layers appear after **every major component** in the encoder (after multi-head attention and after feed-forward networks).

## 5. Feed-Forward Network: Introducing Non-Linearity
<img width="874" height="501" alt="feed_forward" src="https://github.com/user-attachments/assets/a17a81b0-dc7c-4a5d-830a-da4cb75e0556" />

### The Non-Linearity Problem

All operations in multi-head attention are linear transformations (matrix multiplications). Without non-linearity:

1. **Limited learning capacity**: Linear models can only create straight decision boundaries
2. **Layer collapse**: Multiple linear layers can be mathematically reduced to a single layer

**Example of layer collapse**:
```
A₁ = A × W₁ + b₁
A₂ = A₁ × W₂ + b₂ = (A × W₁ + b₁) × W₂ + b₂
   = A × (W₁ × W₂) + (b₁ × W₂ + b₂)
   = A × W' + b'  ← Single equivalent layer!
```

### Architecture

The feed-forward network consists of two dense layers with ReLU activation:

**Structure**:
1. **Expansion layer**: 512 → 2048 dimensions (4× expansion)
   - Increases model capacity to learn complex patterns
   - Like widening the model's perspective
   
2. **Compression layer**: 2048 → 512 dimensions  
   - Returns to original dimensionality for residual connections

**Mathematical representation**:
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

### Why 2048 Dimensions?

The expansion to 2048 neurons allows the model to:
- Capture richer, more nuanced patterns
- Learn hierarchical feature representations
- Achieve better performance (removing FFN layers significantly degrades accuracy)

**Parameter distribution**: Feed-forward networks contain approximately **2 million parameters** versus 1 million in multi-head attention.

## 6. Stacking Encoder Blocks
<img width="974" height="297" alt="stacked_header" src="https://github.com/user-attachments/assets/fa71d8c0-54d5-4136-bc49-55d545c4d257" />

### The Deep Architecture

The original Transformer uses **6 identical encoder blocks** stacked vertically. Each block contains:
- Multi-head attention
- Add & Norm layer
- Feed-forward network  
- Add & Norm layer

### Key Properties

1. **Independent parameters**: Each encoder block has unique, learnable parameters (W₁, b₁, W₂, b₂, etc.)

2. **Positional encoding once**: Only the first encoder receives positional encodings; subsequent blocks receive raw outputs from previous blocks

3. **Consistent dimensionality**: Input and output dimensions remain 512 throughout all blocks (enabling clean residual connections)

4. **Hierarchical learning**:
   - Lower layers (Encoder 1-2): Capture fine-grained, word-level relationships
   - Middle layers (Encoder 3-4): Build phrase and clause structures
   - Upper layers (Encoder 5-6): Extract sentence-level semantics and abstract concepts

## Summary of Information Flow

```
Input Tokens
    ↓
Word Embeddings (512-dim)
    ↓
+ Positional Encodings (512-dim)
    ↓
╔═══════════ Encoder Block 1 ═══════════╗
║  Multi-Head Attention (8 heads)        ║
║            ↓                           ║
║  Add & Norm (Residual + LayerNorm)    ║
║            ↓                           ║
║  Feed-Forward (512→2048→512)          ║
║            ↓                           ║
║  Add & Norm (Residual + LayerNorm)    ║
╚═══════════════════════════════════════╝
    ↓
╔═══════════ Encoder Block 2-6 ═══════╗
║         (Same structure)              ║
╚═══════════════════════════════════════╝
    ↓
Context-Rich Representations
    ↓
(Passed to Decoder)
```

## Detailed Component Breakdown

### Complete Single Encoder Block

```
Input (3 × 512) 
    ↓
[Positional Encoding added to first block only]
    ↓
Multi-Head Attention:
    ├─ Head 1: (3 × 512) → (3 × 64)
    ├─ Head 2: (3 × 512) → (3 × 64)
    ├─ Head 3: (3 × 512) → (3 × 64)
    ├─ ...
    └─ Head 8: (3 × 512) → (3 × 64)
         ↓
    Concatenate: (3 × 512)
         ↓
    Linear: W^o (512 × 512)
         ↓
    Output: (3 × 512)
    ↓
Add & Norm:
    Input + MHA_Output → LayerNorm → (3 × 512)
    ↓
Feed-Forward Network:
    Linear 1: (3 × 512) → (3 × 2048)
         ↓
    ReLU activation
         ↓
    Linear 2: (3 × 2048) → (3 × 512)
    ↓
Add & Norm:
    Previous_Output + FFN_Output → LayerNorm → (3 × 512)
    ↓
Output (3 × 512)
```

## Key Hyperparameters in Original Transformer

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Dimension of embeddings and all layers |
| n_heads | 8 | Number of attention heads |
| d_k = d_v | 64 | Dimension per attention head (512/8) |
| d_ff | 2048 | Inner dimension of FFN |
| N | 6 | Number of encoder blocks |
| dropout | 0.1 | Dropout rate applied throughout |

## Mathematical Formulations

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head₈)W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Feed-Forward Network

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

where:
  μ = mean of features
  σ² = variance of features
  γ, β = learned parameters
```

## Advantages of Encoder Architecture

1. **Parallelization**: Unlike RNNs, all positions are processed simultaneously
2. **Long-range dependencies**: Direct connections between any two positions
3. **Flexible attention patterns**: Multi-head mechanism captures diverse relationships
4. **Stable training**: Residual connections and layer normalization prevent gradient issues
5. **Scalability**: Architecture scales efficiently with model size and data

## Common Applications

- **BERT**: Bidirectional encoder for language understanding
- **RoBERTa**: Optimized BERT training approach
- **ALBERT**: Parameter-efficient encoder
- **Sentence-BERT**: Encoders for semantic similarity
- **Vision Transformer (ViT)**: Encoders adapted for image classification

## Conclusion

The Transformer encoder elegantly combines:
- **Self-attention** for context-aware representations
- **Multi-head mechanisms** for diverse relationship modeling
- **Positional encodings** for sequence order preservation
- **Residual connections** for stable gradient flow
- **Feed-forward networks** for non-linear transformations
- **Layer normalization** for training stability

This architecture revolutionized NLP by enabling parallel processing, capturing long-range dependencies, and achieving unprecedented performance on language understanding tasks.

---

## References

- Vaswani et al. (2017). "Attention Is All You Need"
- Original Transformer paper introducing the encoder-decoder architecture

---

