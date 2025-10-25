# Decoder Architecture in Transformers: A Comprehensive Tutorial

## Introduction

The Transformer decoder is the generative component that produces output sequences. Understanding its architecture reveals how models like GPT generate coherent text and how translation models convert between languages. This tutorial systematically explores each decoder component with special emphasis on masked multi-head attention and cross attention.

---

## 1. Encoder-Decoder Model in Deep Learning

### Traditional RNN-Based Approach

Consider translating English to Spanish: **"I like pizza"** → **"Me gusta la pizza"**

In RNN-based encoder-decoder models:

1. **Encoder Phase**: Input words are processed **sequentially** (one after another)
   - "I" → Hidden State h₁
   - "like" → Hidden State h₂ (contains info from "I" + "like")
   - "pizza" → Hidden State h₃ (contains summarized context of entire sentence)

2. **Decoder Phase**: Uses the final hidden state as context
   - Input: `<START>` + context → Output: "Me"
   - Input: "Me" + context → Output: "gusta"
   - Input: "gusta" + context → Output: "la"
   - Continues until `<END>` token

**Limitations**:
- Slow sequential processing (can't parallelize)
- Long sentences → earlier words are "forgotten" in the final hidden state
- Single summarized context for all decoder steps

---

## 2. Encoder-Decoder in Transformers

### Key Differences from RNNs

**Encoder Side**: Processes **all input words in parallel**
- Input: "I love Apple phones" (all at once)
- Output: 4 contextual vectors (one per word, all 512-dimensional)

**Decoder Side**: Still works sequentially during **inference**

**Inference Process**:
```
Step 1: Input: <START>              → Output: "Me"
Step 2: Input: <START>, "Me"        → Output: "gusta"
Step 3: Input: <START>, "Me", "gusta" → Output: "la"
Step 4: Input: <START>, "Me", "gusta", "la" → Output: "pizza"
Step 5: ...continues until <END>
```

**Why Sequential?** Each predicted word influences the next prediction. We cannot predict all words simultaneously during inference because future words depend on past predictions.

---

## 3. Parallelizing Training in Transformers

### The Training Dilemma

**Question**: If inference is sequential, must training also be sequential?

**Answer**: No! Training can be parallelized because we already know the entire target sequence.

### Training vs Inference Behavior

#### During Training:
- We have the complete target sequence: `<START>, Me, gusta, la, pizza, <END>`
- We use **teacher forcing**: feed correct words (not predicted words) at each step
- We can process all positions in **one forward pass**

**Example Training Step**:
```
Decoder Input:  <START>, Me, gusta, la, pizza
Target Output:  Me, gusta, la, pizza, <END>

Prediction:     la, odio, gusta, pizza, <END>  ← Model predictions (can be wrong)
```

Calculate loss between predictions and targets, then backpropagate. We **discard** wrong predictions and feed correct target words, allowing parallel processing.

### Three Approaches Considered:

1. **Sequential Training**: Slow, mimics inference
2. **Multiple Decoder Instances**: One decoder per word position
   - Problem: Computational cost scales linearly with sequence length
   - 1000 words = 1000 decoder copies!
   
3. **Single Forward Pass with Full Sequence**: ✓ This is the solution!
   - Problem: Creates **data leakage**
   - Solution: **Masked Multi-Head Attention**

---

## 4. Masked Multi-Head Attention ⭐

### The Data Leakage Problem

If we pass the entire target sequence at once, self-attention lets each word "see" all other words, including **future words**.

**Problem Illustration**:
```
To predict "Me":     Should only see: <START>
                     But sees ALL:    <START>, Me, gusta, la, pizza ❌

To predict "gusta":  Should only see: <START>, Me
                     But sees ALL:    <START>, Me, gusta, la, pizza ❌
```

This is **data leakage**: the model accesses information during training that won't be available during inference.

### The Masking Solution

**Masked attention** prevents words from attending to future positions by setting their attention scores to zero.

#### Mathematical Implementation

**Step 1**: Calculate attention scores (before softmax)
```
Input: <START>, Me, gusta, la, pizza (5 × 512)
Query, Key, Value matrices: Each 5 × 64

Attention Scores = Q × K^T / √d_k  → 5 × 5 matrix
```

**Step 2**: Apply the mask
```
Mask Matrix (added before softmax):
[  0,   -∞,   -∞,   -∞,   -∞ ]
[  0,    0,   -∞,   -∞,   -∞ ]
[  0,    0,    0,   -∞,   -∞ ]
[  0,    0,    0,    0,   -∞ ]
[  0,    0,    0,    0,    0 ]
```

Where:
- `0`: Allow attention
- `-∞`: Block attention to future tokens

**Step 3**: After adding mask and applying softmax
```
Attention Weights (after softmax):
[ 1.0,  0,    0,    0,    0   ]  ← <START> only sees itself
[ 0.6,  0.4,  0,    0,    0   ]  ← "Me" sees <START> and itself
[ 0.3,  0.3,  0.4,  0,    0   ]  ← "gusta" sees first 3 tokens
[ 0.2,  0.3,  0.2,  0.3,  0   ]  ← "la" sees first 4 tokens
[ 0.2,  0.2,  0.2,  0.2,  0.2 ]  ← "pizza" sees all tokens
```

**Result**: Softmax converts `-∞` to 0, effectively blocking future information.

### Expanded Equations

For word "Me" (position 2):
```
New_Embedding(Me) = w₁ × Value(<START>) + w₂ × Value(Me) + 0 × Value(gusta) + 0 × Value(la) + 0 × Value(pizza)

where w₁ + w₂ = 1 (from softmax)
```

Future words have zero contribution!

### Multi-Head Masked Attention

Just like encoder, decoder uses **8 attention heads**:
- Each head learns different patterns (temporal, syntactic, semantic)
- All heads apply masking independently
- Outputs concatenated: 8 × 64 = 512 dimensions

---

## 5. Encoder-Decoder Training in Transformers

### Complete Training Flow

```
┌─────────────────────────────────────────┐
│ English Input: "I like pizza"          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ ENCODER (Single Forward Pass)          │
│ Output: 3 contextual vectors (3 × 512) │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ DECODER (Single Forward Pass)          │
│ Input: <START>, Me, gusta, la, pizza   │
│ Uses: Masked Attention + Encoder Output│
│ Output: 5 predictions                   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Compute Loss (all positions at once)   │
│ Target: Me, gusta, la, pizza, <END>    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Backpropagation                         │
│ Update Decoder → Update Encoder        │
└─────────────────────────────────────────┘
```

**Key Point**: Both encoder and decoder process in **one forward pass** during training, but decoder uses masking to prevent cheating.

---

## 6. Positional Encodings

Since masked multi-head attention processes words in parallel, it loses word order information.

**Example**: Without positional info, these are identical:
- "The cat chased the mouse"
- "The mouse chased the cat"

**Solution**: Add positional encodings before the first decoder layer
```
Decoder_Input = Word_Embedding (512-dim) + Positional_Encoding (512-dim)
```

Positional encodings use sinusoidal functions that create predictable patterns, allowing the model to understand relative positions.

**Applied once**: Only before the first decoder block, not between subsequent blocks.

---

## 7. Add & Norm Layer

Applied after **every major component** in the decoder.

### Two Operations:

**1. Residual Connection (Add)**
```
Output = Input + Layer_Output
```

Provides a shortcut path for gradients, preventing information loss in deep networks.

**2. Layer Normalization (Norm)**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

Stabilizes training by:
- Preventing vanishing/exploding gradients
- Ensuring faster convergence

**Usage in Decoder**:
- After masked multi-head attention
- After cross attention
- After feed-forward network

---

## 8. Cross Attention ⭐

### Purpose

Cross attention connects the decoder to the encoder, allowing the decoder to focus on relevant parts of the input sequence.

**Key Question**: How does "gusta" (Spanish) know to attend to "like" (English)?

**Answer**: Cross attention!

### How Cross Attention Differs from Self-Attention

| Component | Self-Attention | Cross Attention |
|-----------|----------------|-----------------|
| Query (Q) | From same input | From **decoder** |
| Key (K)   | From same input | From **encoder** |
| Value (V) | From same input | From **encoder** |

### Mathematical Flow

**Inputs**:
- Decoder output after Add & Norm: 5 × 512 (Spanish words)
- Encoder output: 3 × 512 (English words: "I", "like", "pizza")

**Step 1**: Generate Q, K, V
```
Q = Decoder_Output × W^Q  → 5 × 64  (Query from Spanish words)
K = Encoder_Output × W^K  → 3 × 64  (Key from English words)
V = Encoder_Output × W^V  → 3 × 64  (Value from English words)
```

**Step 2**: Calculate Cross-Attention
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

Q × K^T → 5 × 3 matrix (each Spanish word's similarity to each English word)
```

**Example Attention Scores**:
```
             "I"   "like"  "pizza"
"Me"       [ 0.7    0.2     0.1  ]  ← High attention to "I"
"gusta"    [ 0.1    0.8     0.1  ]  ← High attention to "like"
"la"       [ 0.3    0.4     0.3  ]  ← Distributed attention
"pizza"    [ 0.1    0.1     0.8  ]  ← High attention to "pizza"
<END>      [ 0.2    0.3     0.5  ]
```

**Step 3**: Generate new contextual embeddings
```
New_Embedding(gusta) = 0.1 × Value(I) + 0.8 × Value(like) + 0.1 × Value(pizza)
```

"gusta" now has strong contextual connection to "like"!

### Advantages Over RNN Approach

**RNN Problem**: Single summarized context vector for entire sentence
- Long sentences → early words forgotten
- All decoder words use same fixed context

**Transformer Solution**: Dynamic, word-specific context
- "gusta" focuses on "like"
- "pizza" focuses on "pizza"
- 100-word sentence? No problem! All 100 encoder outputs remain accessible

### Cross Attention in Multimodal Learning

Cross attention powers **multimodal AI** (e.g., ChatGPT with images):

**Example**:
- Query: Text embeddings (from user question)
- Key & Value: Image embeddings (from uploaded picture)
- Result: Text attends to relevant image regions

This mimics human perception: integrating visual and linguistic information simultaneously.

---

## 9. Feed-Forward Network

After cross attention, another Add & Norm layer, then the feed-forward network.

### Architecture
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

Dimensions: 512 → 2048 → 512
```

### Two Key Functions:

**1. Introduce Non-Linearity**
- Masked attention and cross attention use only linear transformations
- Without non-linearity, multiple layers collapse into one
- ReLU activation breaks linearity

**2. Expand Model Capacity**
- 4× expansion (512 → 2048) captures richer patterns
- Similar to fully connected layers in CNNs
- Contains ~2 million parameters vs. ~1 million in attention

**Why return to 512?** Maintains consistent dimensions for residual connections.

---

## 10. Stacking Decoder Blocks

### Architecture

**6 decoder blocks** stacked vertically (in original Transformer):

```
Positional Encoding (applied once)
    ↓
╔══════════════════════════════════════╗
║ Decoder Block 1                      ║
║  ├─ Masked Multi-Head Attention      ║
║  ├─ Add & Norm                       ║
║  ├─ Cross Attention (with Encoder)   ║
║  ├─ Add & Norm                       ║
║  ├─ Feed-Forward Network             ║
║  └─ Add & Norm                       ║
╚══════════════════════════════════════╝
    ↓
╔══════════════════════════════════════╗
║ Decoder Block 2-6                    ║
║ (Same structure, different params)   ║
╚══════════════════════════════════════╝
    ↓
Final Prediction Layer
```

### Key Properties:

1. **Independent Parameters**: Each block has unique learned weights
2. **No Positional Encoding Between Blocks**: Applied only once at the start
3. **Consistent Dimensionality**: Always 512 throughout
4. **Hierarchical Learning**:
   - Lower blocks: Word-level patterns
   - Middle blocks: Phrase structures
   - Upper blocks: Sentence-level semantics

---

## 11. Final Prediction Layer

### Converting Embeddings to Words

**Input**: 512-dimensional vector from final decoder block  
**Output**: Spanish word from vocabulary

**Process**:

**Step 1**: Linear Transformation
```
Logits = Decoder_Output × W_out + b

Where:
  Decoder_Output: 512-dim
  W_out: 512 × 100,000 (vocab size)
  Logits: 100,000-dim vector
```

**Step 2**: Softmax to Probabilities
```
Probabilities = softmax(Logits)  → Sum to 1.0

Example:
Position 1 (<START>):  0.001
Position 2 (<END>):    0.002
Position 573 (Me):     0.856  ← Highest probability
Position 891 (gusta):  0.023
...
Position 99,999:       0.0001
```

**Step 3**: Prediction = argmax(Probabilities)
```
Prediction = "Me" (position 573 has highest probability)
```

### Multiple Outputs

If decoder receives 5 input tokens → produces 5 predictions (one per position).

During training: Compare all 5 predictions with targets simultaneously, compute loss, backpropagate.

---

## 12. Decoder During Inference

### Sequential Generation

Unlike training, inference **must be sequential**:

```
Step 1:
  Input:  [<START>]
  Output: [Me]           ← Keep only this

Step 2:
  Input:  [<START>, Me]
  Output: [X, gusta]     ← Discard first, keep second

Step 3:
  Input:  [<START>, Me, gusta]
  Output: [X, X, la]     ← Discard first two, keep third

Step 4:
  Input:  [<START>, Me, gusta, la]
  Output: [X, X, X, pizza]  ← Keep only last

Step 5:
  Input:  [<START>, Me, gusta, la, pizza]
  Output: [X, X, X, X, <END>]  ← Stop generation
```

**Why Discard Earlier Outputs?** We only need the prediction for the **newly added token**.

### Masking During Inference

**Critical Point**: Masking is **still applied** during inference!

**Why?**
- Cannot change architecture between training and inference
- Mask layer has trained parameters
- Disabling it would confuse the model

Even though we process tokens sequentially, the mask ensures consistency with training behavior.

### Training vs Inference Summary

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input** | Full target sequence | Previously generated tokens |
| **Processing** | Parallel (single forward pass) | Sequential (one token at a time) |
| **Masking** | Applied | Applied |
| **Speed** | Fast (parallelized) | Slower (sequential) |
| **Purpose** | Learn parameters | Generate new text |

---

## Complete Information Flow

```
┌──────────────────────────────────────────────────┐
│ Input: Target Sequence (Spanish words)          │
│ <START>, Me, gusta, la, pizza                   │
└──────────────┬───────────────────────────────────┘
               │
               ▼
     [+ Positional Encodings] (512-dim)
               │
               ▼
┌──────────────────────────────────────────────────┐
│ Masked Multi-Head Attention (8 heads)           │
│ - Prevents future token visibility              │
│ - Learns temporal patterns                      │
└──────────────┬───────────────────────────────────┘
               │
               ▼
     [Add & Norm: Residual + LayerNorm]
               │
               ▼
┌──────────────────────────────────────────────────┐
│ Cross Attention (8 heads)                       │
│ - Q from decoder, K & V from encoder            │
│ - Connects Spanish to English                   │
│ - Dynamic context per word                      │
└──────────────┬───────────────────────────────────┘
               │
               ▼
     [Add & Norm: Residual + LayerNorm]
               │
               ▼
┌──────────────────────────────────────────────────┐
│ Feed-Forward Network                             │
│ - 512 → 2048 → 512                              │
│ - Introduces non-linearity (ReLU)              │
│ - Expands model capacity                        │
└──────────────┬───────────────────────────────────┘
               │
               ▼
     [Add & Norm: Residual + LayerNorm]
               │
               ▼
    [Repeat 6 times: Decoder Blocks 1-6]
               │
               ▼
┌──────────────────────────────────────────────────┐
│ Linear Layer: 512 × 100,000                     │
│ Softmax: Convert to probabilities               │
│ Prediction: argmax (highest probability word)   │
└──────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Masked Multi-Head Attention**: Prevents data leakage by blocking future tokens, enabling parallel training while maintaining causal dependencies

2. **Cross Attention**: Bridges encoder-decoder, allowing dynamic focus on relevant input positions rather than fixed context vectors

3. **Training vs Inference**: Transformers behave differently—parallel training with masking vs. sequential inference with autoregressive generation

4. **Residual Connections**: Stabilize deep networks, prevent gradient issues, maintain information flow

5. **Architecture Consistency**: Masking and all trained parameters remain active during inference to maintain model behavior

6. **Multimodal Capability**: Cross attention enables integration of different data types (text, image, audio)

---

## Conclusion

The Transformer decoder elegantly solves the sequence generation problem through:
- **Masked attention** for causal dependencies without sequential training
- **Cross attention** for dynamic, context-aware input integration
- **Residual connections** for stable training in deep networks
- **Feed-forward networks** for non-linearity and capacity
- **Careful design** balancing training efficiency with inference correctness

This architecture revolutionized generative AI, powering modern language models like GPT, translation systems, and multimodal AI assistants.

---

