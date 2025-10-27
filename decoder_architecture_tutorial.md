# Decoder Architecture in Transformers: A Comprehensive Tutorial
Suurce: https://youtu.be/DFqWPwF0OH0?si=luQ3WN4H6Xh3LSpa

## Introduction

The Transformer decoder is the generative component that produces output sequences. Understanding its architecture reveals how models like GPT generate coherent text and how translation models convert between languages. This tutorial systematically explores each decoder component with special emphasis on masked multi-head attention and cross attention, using **English to Bangla translation** as our primary example.

---

## 1. Encoder-Decoder Model in Deep Learning

### Traditional RNN-Based Approach

Consider translating English to Bangla: **"I like pizza"** → **"আমি পিজ্জা পছন্দ করি"**

In RNN-based encoder-decoder models:

1. **Encoder Phase**: Input words are processed **sequentially** (one after another)
   - "I" → Hidden State h₁
   - "like" → Hidden State h₂ (contains info from "I" + "like")
   - "pizza" → Hidden State h₃ (contains summarized context of entire sentence)

2. **Decoder Phase**: Uses the final hidden state as context
   - Input: `<START>` + context → Output: "আমি" (I)
   - Input: "আমি" + context → Output: "পিজ্জা" (pizza)
   - Input: "পিজ্জা" + context → Output: "পছন্দ" (like)
   - Input: "পছন্দ" + context → Output: "করি" (do)
   - Continues until `<END>` token

**Limitations**:
- Slow sequential processing (can't parallelize)
- Long sentences → earlier words are "forgotten" in the final hidden state
- Single summarized context for all decoder steps

---

## 2. Encoder-Decoder in Transformers

### Key Differences from RNNs

**Encoder Side**: Processes **all input words in parallel**
- Input: "I like pizza" (all at once)
- Output: 3 contextual vectors (one per word, all 512-dimensional)

**Decoder Side**: Still works sequentially during **inference**

**Inference Process** (English → Bangla):
```
Step 1: Input: <START>                    → Output: "আমি"
Step 2: Input: <START>, "আমি"             → Output: "পিজ্জা"
Step 3: Input: <START>, "আমি", "পিজ্জা"  → Output: "পছন্দ"
Step 4: Input: <START>, "আমি", "পিজ্জা", "পছন্দ" → Output: "করি"
Step 5: ...continues until <END>
```

**Why Sequential?** Each predicted word influences the next prediction. We cannot predict all words simultaneously during inference because future words depend on past predictions.

**Important Note**: Word order differs between English and Bangla:
- English: "I like pizza" (Subject-Verb-Object)
- Bangla: "আমি পিজ্জা পছন্দ করি" (Subject-Object-Verb)

The decoder learns these structural differences through cross attention.

---

## 3. Parallelizing Training in Transformers

### The Training Dilemma

**Question**: If inference is sequential, must training also be sequential?

**Answer**: No! Training can be parallelized because we already know the entire target sequence.

### Training vs Inference Behavior

#### During Training:
- We have the complete target sequence: `<START>, আমি, পিজ্জা, পছন্দ, করি, <END>`
- We use **teacher forcing**: feed correct words (not predicted words) at each step
- We can process all positions in **one forward pass**

**Example Training Step**:
```
Decoder Input:  <START>, আমি, পিজ্জা, পছন্দ, করি
Target Output:  আমি, পিজ্জা, পছন্দ, করি, <END>

Prediction:     আমি, খাবার, পছন্দ, করি, <END>  ← Model predictions (some wrong)
                     ↑ Wrong! Should be "পিজ্জা"
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

**Problem Illustration** (Bangla sequence):
```
To predict "আমি":     Should only see: <START>
                      But sees ALL:    <START>, আমি, পিজ্জা, পছন্দ, করি ❌

To predict "পিজ্জা":  Should only see: <START>, আমি
                      But sees ALL:    <START>, আমি, পিজ্জা, পছন্দ, করি ❌

To predict "পছন্দ":   Should only see: <START>, আমি, পিজ্জা
                      But sees ALL:    <START>, আমি, পিজ্জা, পছন্দ, করি ❌
```

This is **data leakage**: the model accesses information during training that won't be available during inference.

### The Masking Solution

**Masked attention** prevents words from attending to future positions by setting their attention scores to zero.

#### Mathematical Implementation

*The original self-attention:*
<img width="960" height="480" alt="image" src="https://github.com/user-attachments/assets/e658fa0b-0b17-421f-96ec-76879d704561" />

**Step 1**: Calculate attention scores (before softmax)
```
Input: <START>, আমি, পিজ্জা, পছন্দ, করি (5 × 512)
Query, Key, Value matrices: Each 5 × 64

Attention Scores = Q × K^T / √d_k  → 5 × 5 matrix
```

**Step 2**: Apply the mask

*The modification (masked attention) over self-attention:*
<img width="960" height="480" alt="image" src="https://github.com/user-attachments/assets/dd8a92a3-159b-4134-b3d1-24d27238a605" />


```
Mask Matrix (added before softmax):
              <START>  আমি   পিজ্জা  পছন্দ  করি
<START>    [    0,    -∞,    -∞,    -∞,   -∞  ]
আমি        [    0,     0,    -∞,    -∞,   -∞  ]
পিজ্জা     [    0,     0,     0,    -∞,   -∞  ]
পছন্দ      [    0,     0,     0,     0,   -∞  ]
করি        [    0,     0,     0,     0,    0  ]
```

Where:
- `0`: Allow attention (can see this token)
- `-∞`: Block attention to future tokens

**Visualization**:
```
Upper triangular part = -∞ (future tokens blocked)
Lower triangular part + diagonal = 0 (past + current tokens allowed)
```

**Step 3**: After adding mask and applying softmax
```
Attention Weights (after softmax):
              <START>  আমি   পিজ্জা  পছন্দ  করি
<START>    [   1.0,    0,     0,     0,    0   ]  ← Only sees itself
আমি        [   0.6,  0.4,    0,     0,    0   ]  ← Sees <START> and itself
পিজ্জা     [   0.3,  0.3,  0.4,    0,    0   ]  ← Sees first 3 tokens
পছন্দ      [   0.2,  0.3,  0.2,  0.3,   0   ]  ← Sees first 4 tokens
করি        [   0.2,  0.2,  0.2,  0.2,  0.2 ]  ← Sees all tokens
```

**Result**: Softmax converts `-∞` to 0, effectively blocking future information.

### Detailed Mathematical Explanation

**Why -∞ in the mask?**

The softmax function is:
```
softmax(x_i) = e^(x_i) / Σ(e^(x_j))
```

When x_i = -∞:
```
e^(-∞) = 0
```

So any score with -∞ becomes 0 after softmax, effectively removing that connection.

### Expanded Equations

For word "পিজ্জা" (position 3):
```
New_Embedding(পিজ্জা) = w₁ × Value(<START>) + w₂ × Value(আমি) + w₃ × Value(পিজ্জা) 
                        + 0 × Value(পছন্দ) + 0 × Value(করি)

where w₁ + w₂ + w₃ = 1 (from softmax normalization)
```

Future words "পছন্দ" and "করি" have **zero contribution**!

**Complete Self-Attention with Masking**:
```
1. Linear Transformations:
   Q = Input × W^Q  (5 × 512) × (512 × 64) = 5 × 64
   K = Input × W^K  (5 × 512) × (512 × 64) = 5 × 64
   V = Input × W^V  (5 × 512) × (512 × 64) = 5 × 64

2. Calculate Scores:
   Scores = (Q × K^T) / √64 = 5 × 5 matrix

3. Apply Mask:
   Masked_Scores = Scores + Mask_Matrix

4. Softmax:
   Attention_Weights = softmax(Masked_Scores)

5. Final Output:
   Output = Attention_Weights × V = 5 × 64
```

### Multi-Head Masked Attention

Just like encoder, decoder uses **8 attention heads**:
- Each head learns different patterns (temporal, syntactic, semantic)
- All heads apply masking independently
- Each head output: 5 × 64
- Outputs concatenated: 8 × 64 = 512 dimensions
- Final linear transformation: W^O (512 × 512)

**Why multiple heads for Bangla?**
- **Head 1**: Subject-verb agreement (আমি with করি)
- **Head 2**: Object relationships (পিজ্জা as object)
- **Head 3**: Verb conjugation patterns
- **Head 4**: Postposition markers (common in Bangla)
- **Head 5-8**: Other linguistic patterns

---

## 5. Encoder-Decoder Training in Transformers

### Complete Training Flow (English → Bangla)

```
┌───────────────────────────────────────────┐
│ English Input: "I like pizza"            │
│ Tokenized: ["I", "like", "pizza"]       │
└────────────┬──────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────┐
│ ENCODER (Single Forward Pass)            │
│ - Processes all English words in parallel│
│ Output: 3 contextual vectors (3 × 512)   │
│   Vector₁ for "I"                        │
│   Vector₂ for "like"                     │
│   Vector₃ for "pizza"                    │
└────────────┬──────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────┐
│ DECODER (Single Forward Pass)            │
│ Input: <START>, আমি, পিজ্জা, পছন্দ, করি  │
│ Uses:                                     │
│  - Masked Attention (prevents cheating)  │
│  - Cross Attention (connects to encoder) │
│ Output: 5 predictions                     │
│   Pred₁, Pred₂, Pred₃, Pred₄, Pred₅     │
└────────────┬──────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────┐
│ Compute Loss (all positions at once)     │
│ Target: আমি, পিজ্জা, পছন্দ, করি, <END>   │
│ Compare predictions with targets          │
│ Loss = CrossEntropy(Predictions, Targets) │
└────────────┬──────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────┐
│ Backpropagation                           │
│ 1. Update Decoder weights                 │
│ 2. Gradients flow back to Encoder        │
│ 3. Update Encoder weights                 │
└───────────────────────────────────────────┘
```

**Key Point**: Both encoder and decoder process in **one forward pass** during training, but decoder uses masking to prevent cheating.

**Training Example Iteration**:
```
Epoch 1:
  Encoder Input:  "I", "like", "pizza"
  Decoder Input:  <START>, আমি, পিজ্জা, পছন্দ, করি
  Predictions:    খাবার, পছন্দ, আমি, করি, <END>  ← Wrong!
  Loss: High
  
Epoch 100:
  Encoder Input:  "I", "like", "pizza"
  Decoder Input:  <START>, আমি, পিজ্জা, পছন্দ, করি
  Predictions:    আমি, পিজ্জা, পছন্দ, করি, <END>  ← Correct!
  Loss: Low
```

---

## 6. Positional Encodings

Since masked multi-head attention processes words in parallel, it loses word order information.

**Example - Critical in Bangla**: Word order changes meaning:
- "আমি পছন্দ করি" (I like) - Subject before verb
- "করি পছন্দ আমি" - Grammatically incorrect, meaningless

**Solution**: Add positional encodings before the first decoder layer
```
Decoder_Input = Word_Embedding (512-dim) + Positional_Encoding (512-dim)
```

**Positional Encoding Values** (using sine and cosine functions):
```
For position 0 (<START>):  [0.000, 1.000, 0.000, 1.000, ...]
For position 1 (আমি):      [0.841, 0.540, 0.010, 0.999, ...]
For position 2 (পিজ্জা):   [0.909, -0.416, 0.020, 0.998, ...]
For position 3 (পছন্দ):    [0.141, -0.989, 0.030, 0.995, ...]
For position 4 (করি):      [-0.757, -0.653, 0.039, 0.992, ...]
```

Each position gets a unique pattern using:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Applied once**: Only before the first decoder block, not between subsequent blocks.

**Why this matters for Bangla**: Bangla uses SOV (Subject-Object-Verb) order, while English uses SVO (Subject-Verb-Object). Positional encodings help the model learn this reordering.

---

## 7. Add & Norm Layer

Applied after **every major component** in the decoder.

### Two Operations:

**1. Residual Connection (Add)**
```
Output = Input + Layer_Output
```

Provides a shortcut path for gradients, preventing information loss in deep networks.

**Example**:
```
Input to Masked Attention: Z₀ (5 × 512)
Output from Masked Attention: Z₀' (5 × 512)
Add & Norm Input: Z₀ + Z₀' (element-wise addition)
```

**2. Layer Normalization (Norm)**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

where:
  μ = mean across features (for each word separately)
  σ² = variance across features
  ε = 1e-5 (numerical stability)
  γ, β = learned scaling and shifting parameters
```

**Stabilizes training by**:
- Preventing vanishing/exploding gradients
- Ensuring faster convergence
- Normalizing activations to standard range

**Usage in Decoder**:
- After masked multi-head attention
- After cross attention  
- After feed-forward network

**Total**: 3 Add & Norm layers per decoder block

---

## 8. Cross Attention ⭐

### Purpose

Cross attention connects the decoder to the encoder, allowing the decoder to focus on relevant parts of the input sequence.

**Key Question**: How does "পছন্দ" (like in Bangla) know to attend to "like" (in English) when they appear in different positions?

**Answer**: Cross attention creates dynamic connections!

### English-Bangla Translation Challenge

**Word Order Difference**:
```
English: "I"    "like"  "pizza"  (SVO order)
         ↓       ↓       ↓
Bangla:  "আমি"  "পিজ্জা" "পছন্দ"  "করি"  (SOV order)
         I      pizza   like    do
```

Cross attention learns these complex mappings automatically!

### How Cross Attention Differs from Self-Attention

| Component | Self-Attention | Cross Attention |
|-----------|----------------|-----------------|
| Query (Q) | From same input | From **decoder** (Bangla words) |
| Key (K)   | From same input | From **encoder** (English words) |
| Value (V) | From same input | From **encoder** (English words) |
| Purpose | Relate words within same sequence | Relate words **across** sequences |

### Mathematical Flow

**Inputs**:
- **Decoder output** after Add & Norm: 5 × 512
  - Rows: `<START>, আমি, পিজ্জা, পছন্দ, করি`
  
- **Encoder output**: 3 × 512
  - Rows: `I, like, pizza`

**Step 1**: Generate Q, K, V
```
Q = Decoder_Output × W^Q  → 5 × 64  (Query from Bangla words)
K = Encoder_Output × W^K  → 3 × 64  (Key from English words)
V = Encoder_Output × W^V  → 3 × 64  (Value from English words)
```

**Step 2**: Calculate Cross-Attention Scores
```
Attention_Scores = (Q × K^T) / √d_k

Shape: (5 × 64) × (64 × 3) = 5 × 3

Result Matrix:
             "I"    "like"  "pizza"
<START>    [ s₁₁     s₁₂     s₁₃  ]
আমি        [ s₂₁     s₂₂     s₂₃  ]
পিজ্জা     [ s₃₁     s₃₂     s৩৩  ]
পছন্দ      [ s₄₁     s₄₂     s৪৩  ]
করি        [ s₅₁     s₅₂     s৫৩  ]
```

Each score represents similarity between a Bangla word and an English word.

**Step 3**: Apply Softmax (row-wise)
```
Attention_Weights = softmax(Attention_Scores)

Example Attention Weights:
             "I"    "like"  "pizza"   (Sum per row = 1.0)
<START>    [ 0.33   0.33    0.34  ]  ← Distributed attention
আমি        [ 0.85   0.10    0.05  ]  ← High attention to "I"
পিজ্জা     [ 0.05   0.05    0.90  ]  ← High attention to "pizza"
পছন্দ      [ 0.10   0.80    0.10  ]  ← High attention to "like"
করি        [ 0.15   0.70    0.15  ]  ← High attention to "like" (verb helper)
```

**Step 4**: Generate new contextual embeddings
```
Output = Attention_Weights × V

New_Embedding(পছন্দ) = 0.10 × Value(I) + 0.80 × Value(like) + 0.10 × Value(pizza)
```

Now "পছন্দ" (like in Bangla) has **strong contextual connection** to "like" (in English), even though they're in different positions!

### Complete Cross-Attention Equation

```
CrossAttention(Q, K, V) = softmax((Q × K^T) / √d_k) × V

where:
  Q from Decoder (Bangla context)
  K, V from Encoder (English context)
```

### Visualization of Attention Patterns

```
English:  I ----→ আমি    (Subject to subject)
          |
          like --→ পছন্দ  (Verb to verb, despite position change)
          |     ↗
          pizza → পিজ্জা  (Object to object)
```

### Multi-Head Cross Attention

**8 parallel attention heads**, each learning different alignment patterns:

**Head 1**: Subject alignment
```
"I" ↔ "আমি" (strong connection)
```

**Head 2**: Object alignment
```
"pizza" ↔ "পিজ্জা" (strong connection)
```

**Head 3**: Verb alignment
```
"like" ↔ "পছন্দ করি" (connects to verb phrase)
```

**Head 4**: Syntactic structure
```
Learns SVO → SOV reordering patterns
```

**Heads 5-8**: Other linguistic patterns
- Tense markers
- Article handling (English has "a/an/the", Bangla doesn't)
- Number agreement
- Formality levels

### Advantages Over RNN Approach

**RNN Problem**: Single summarized context vector
```
Encoder → [Hidden State] → Decoder
          (All info compressed into one vector)
```

For "I like pizza":
- Final hidden state tries to encode all information
- Early words "I" may be forgotten
- Same fixed context for all decoder words

**Transformer Solution**: Dynamic, word-specific context
```
Encoder → [Vector_I, Vector_like, Vector_pizza] → Decoder
          (Separate representation for each word)
```

Benefits:
- "আমি" focuses primarily on "I"
- "পিজ্জা" focuses primarily on "pizza"  
- "পছন্দ" focuses primarily on "like"
- No information loss even in long sentences

**Long Sentence Example**:
```
English (100 words) → Encoder → 100 contextual vectors
                                      ↓
                      All 100 remain accessible to decoder
                                      ↓
                      Bangla decoder attends to relevant ones
```

### Cross Attention in Multimodal Learning

Cross attention powers **multimodal AI** systems that process different data types:

**Example 1 - Image Captioning** (Bangla):
```
Query (Q):  Text decoder generating: "এই ছবিতে..."
Key (K):    Image patch embeddings from CNN
Value (V):  Image patch features

Result: Text attends to relevant image regions
Output: "এই ছবিতে একটি বিড়াল আছে" (There's a cat in this image)
```

**Example 2 - Visual Question Answering**:
```
Image: [Picture of pizza]
Question: "এটি কী?" (What is this?)
Cross Attention: Question tokens attend to image features
Answer: "এটি একটি পিজ্জা" (This is a pizza)
```

**Example 3 - Audio-Text Alignment**:
```
Audio: Bangla speech waveform
Text: Transcript being generated
Cross Attention: Text tokens attend to audio features
```

This mimics **human perception**: We integrate visual, linguistic, and auditory information simultaneously.

---

## 9. Feed-Forward Network

After cross attention, another Add & Norm layer, then the feed-forward network.

### Architecture
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

Dimensions: 512 → 2048 → 512
```

**Applied to each position independently** (same network, different data).

### Two Key Functions:

**1. Introduce Non-Linearity**

All attention operations (masked and cross) are linear:
```
Attention = softmax(QK^T)V  ← No activation function!
```

Without FFN:
```
Layer1 = A × W₁
Layer2 = Layer1 × W₂ = (A × W₁) × W₂ = A × (W₁W₂)
       = A × W'  ← Collapsed to single layer!
```

**ReLU breaks this**:
```
FFN = ReLU(xW₁ + b₁)W₂ + b₂
    ↑ Non-linear activation prevents collapse
```

**2. Expand Model Capacity**

The 4× expansion (512 → 2048) allows learning complex patterns:

**Example for Bangla**:
```
Input (512-dim): "পছন্দ করি" (like do - verb phrase)

Hidden Layer (2048-dim): 
  - Neurons 1-256: Detect verb root "পছন্দ"
  - Neurons 257-512: Detect auxiliary "করি"
  - Neurons 513-768: Recognize present tense
  - Neurons 769-1024: Identify first person
  - Neurons 1025-2048: Complex morphological patterns

Output (512-dim): Rich representation encoding all above features
```

**Parameter Count**: ~2 million in FFN vs ~1 million in attention

### Why Return to 512 Dimensions?

Maintains consistent dimensions for:
- Residual connections (need matching sizes for addition)
- Next decoder block input
- Cross attention compatibility

### Detailed Architecture

```
For each position (e.g., "পছন্দ"):

Input: x (1 × 512)

Layer 1: 
  z = xW₁ + b₁  
  W₁: (512 × 2048)
  b₁: (2048)
  z: (1 × 2048)

Activation:
  a = ReLU(z) = max(0, z)
  a: (1 × 2048)

Layer 2:
  output = aW₂ + b₂
  W₂: (2048 × 512)
  b₂: (512)
  output: (1 × 512)
```

**Position-wise**: Same FFN applied to all 5 Bangla words, but independently:
```
FFN(আমি), FFN(পিজ্জা), FFN(পছন্দ), FFN(করি) - all use same weights
```

---

## 10. Stacking Decoder Blocks

### Architecture

**6 decoder blocks** stacked vertically (in original Transformer):

```
Bangla Input: <START>, আমি, পিজ্জা, পছন্দ, করি
    ↓
Positional Encoding (applied once)
    ↓
╔════════════════════════════════════════════╗
║ Decoder Block 1                            ║
║  ┌──────────────────────────────────────┐  ║
║  │ Masked Multi-Head Attention (8 heads)│  ║
║  │ - Prevents future token visibility   │  ║
║  └──────────────────────────────────────┘  ║
║    ↓                                        ║
║  [Add & Norm]                               ║
║    ↓                                        ║
║  ┌──────────────────────────────────────┐  ║
║  │ Cross Attention (8 heads)            │  ║
║  │ - Attends to English encoder output  │  ║
║  │ - Q from Bangla, K&V from English    │  ║
║  └──────────────────────────────────────┘  ║
║    ↓                                        ║
║  [Add & Norm]                               ║
║    ↓                                        ║
║  ┌──────────────────────────────────────┐  ║
║  │ Feed-Forward Network                 │  ║
║  │ - 512 → 2048 → 512                  │  ║
║  │ - ReLU activation                    │  ║
║  └──────────────────────────────────────┘  ║
║    ↓                                        ║
║  [Add & Norm]                               ║
╚════════════════════════════════════════════╝
    ↓
╔════════════════════════════════════════════╗
║ Decoder Blocks 2-6                         ║
║ (Same structure, different parameters)     ║
╚════════════════════════════════════════════╝
    ↓
Final Prediction Layer
```

### Key Properties:

**1. Independent Parameters**
```
Block 1: W₁^Q, W₁^K, W₁^V, W₁^O, FFN₁ weights
Block 2: W₂^Q, W₂^K, W₂^V, W₂^O, FFN₂ weights
...
Block 6: W₆^Q, W₆^K, W₆^V, W₆^O, FFN₆ weights

All different, all learned during training
```

**2. No Positional Encoding Between Blocks**
```
Block 1 output → Block 2 input (direct, no processing)
Block 2 output → Block 3 input (direct, no processing)
...
```

**3. Consistent Dimensionality**: Always 512 throughout
```
Each block: Input (5 × 512) → Output (5 × 512)
```

**4. Hierarchical Learning** (What each layer learns for Bangla):

**Lower Blocks (1-2)**:
- Word-level morphology
- Character patterns in Bangla script
- Basic word relationships
- Example: "করি" recognized as verb ending

**Middle Blocks (3-4)**:
- Phrase structures (verb phrases like "পছন্দ করি")
- Grammatical dependencies
- Postposition usage (common in Bangla)
- Example: Connecting "পিজ্জা" as object with "পছন্দ" as verb

**Upper Blocks (5-6)**:
- Sentence-level semantics
- Overall meaning and intent
- Cross-lingual alignment refinement
- Example: Understanding entire sentiment "I like pizza"

### Information Flow Through Blocks

```
Input: আমি (I)

Block 1: Basic word recognition + position
  "This is a first-person pronoun at position 1"

Block 2: Local context
  "আমি is the subject, followed by object পিজ্জা"

Block 3: Verb phrase understanding
  "Subject আমি relates to verb phrase পছন্দ করি"

Block 4: Cross-lingual alignment strengthening
  "আমি strongly maps to English 'I'"

Block 5: Syntactic structure
  "SOV sentence structure, subject role confirmed"

Block 6: Semantic completeness
  "Complete understanding: subject expressing preference"
```

Each block refines the representation progressively.

---

## 11. Final Prediction Layer

### Converting Embeddings to Words

After 6 decoder blocks, we have rich 512-dimensional representations. Now we need to convert them to actual Bangla words.

**Input**: 512-dimensional vector for each position  
**Output**: Bangla word from vocabulary

### Bangla Vocabulary Example

```
Vocabulary Size: 100,000 Bangla words/subwords

Position    Word/Token
0           <START>
1           <END>
2           <PAD>
3           <UNK>
4           আমি (I)
5           তুমি (you)
6           সে (he/she)
...
573         পিজ্জা (pizza)
...
1247        পছন্দ (like/prefer)
...
3891        করি (do - 1st person)
...
99,999      [last vocabulary item]
```

**Note**: Bangla uses more vocabulary items due to:
- Rich morphology (verb conjugations)
- Compound words
- Different formality levels (তুমি vs আপনি for "you")

### The Prediction Process

**Step 1**: Linear Transformation
```
Logits = Decoder_Output × W_out + b_out

Where:
  Decoder_Output: 1 × 512 (for one position)
  W_out: 512 × 100,000
  b_out: 100,000
  Logits: 1 × 100,000 (raw scores for each vocab word)
```

**Step 2**: Softmax to Probabilities
```
Probabilities = softmax(Logits)

softmax(x_i) = e^(x_i) / Σ(e^(x_j))

Result: 100,000 probabilities that sum to 1.0
```

**Step 3**: Example Prediction for Position 1

```
After processing <START> token:

Logits (raw scores):
  Position 0 (<START>):   -5.2
  Position 1 (<END>):     -8.1
  Position 4 (আমি):       12.7  ← Highest!
  Position 5 (তুমি):      -2.3
  Position 573 (পিজ্জা):  -3.1
  Position 1247 (পছন্দ):  -4.5
  ...

After Softmax (probabilities):
  Position 0 (<START>):   0.0001
  Position 1 (<END>):     0.00001
  Position 4 (আমি):       0.9234  ← Highest probability!
  Position 5 (তুমি):      0.0156
  Position 573 (পিজ্জা):  0.0089
  Position 1247 (পছন্দ):  0.0021
  ...
  Sum = 1.0000

Prediction = argmax(Probabilities) = আমি
```

**Step 4**: Prediction Selection
```
Predicted_Word = Vocabulary[argmax(Probabilities)]
               = Vocabulary[4]
               = "আমি"
```

### Multiple Positions Example

```
Decoder Input (5 tokens): <START>, আমি, পিজ্জা, পছন্দ, করি
Decoder Output: 5 × 512 matrix

Position 1 prediction:
  Logits: 5 × 100,000 matrix
  Softmax: Row 1 → Prediction: আমি

Position 2 prediction:
  Softmax: Row 2 → Prediction: পিজ্জা

Position 3 prediction:
  Softmax: Row 3 → Prediction: পছন্দ

Position 4 prediction:
  Softmax: Row 4 → Prediction: করি

Position 5 prediction:
  Softmax: Row 5 → Prediction: <END>
```

### Loss Calculation During Training

```
Target Sequence:  [আমি, পিজ্জা, পছন্দ, করি, <END>]
Predicted Probs:  [P₁,  P₂,    P₃,   P₄,  P₅]

Cross-Entropy Loss:
Loss = -Σ log(P_target_i)

For position 1:
  If target is আমি (index 4) and P(আমি) = 0.9234
  Loss₁ = -log(0.9234) = 0.0796

For position 2:
  If target is পিজ্জা (index 573) and P(পিজ্জা) = 0.8501
  Loss₂ = -log(0.8501) = 0.1625

Total_Loss = (Loss₁ + Loss₂ + Loss₃ + Loss₄ + Loss₅) / 5
```

Lower loss = Better predictions!

### Visualization of Prediction Layer

```
┌─────────────────────────────────────────┐
│ Decoder Block 6 Output                  │
│ Shape: 5 × 512                          │
│ [আমি_vec, পিজ্জা_vec, পছন্দ_vec, করি_vec, END_vec] │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Linear Layer (W: 512 × 100,000)        │
│ For each position independently         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Logits: 5 × 100,000                     │
│ Raw scores for each vocabulary word     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Softmax (applied row-wise)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Probabilities: 5 × 100,000              │
│ Each row sums to 1.0                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Argmax (select highest probability)     │
│ Final Predictions: [আমি, পিজ্জা, পছন্দ, করি, <END>] │
└─────────────────────────────────────────┘
```

---

## 12. Decoder During Inference

### Sequential Generation

Unlike training (parallel), inference **must be sequential**:

```
═══════════════════════════════════════════
STEP 1: Generate First Word
═══════════════════════════════════════════

Input to Decoder:  [<START>]
                    ↓
Encoder provides:  [I_vec, like_vec, pizza_vec]
                    ↓
Decoder processes: 1 position
                    ↓
Output predictions: [Pred₁]
                    ↓
Select Pred₁:      আমি (probability: 0.9234)
Keep: আমি, Discard: None


═══════════════════════════════════════════
STEP 2: Generate Second Word
═══════════════════════════════════════════

Input to Decoder:  [<START>, আমি]
                    ↓
Encoder provides:  [I_vec, like_vec, pizza_vec]
                    ↓
Decoder processes: 2 positions
                    ↓
Output predictions: [Pred₁, Pred₂]
                    ↓
Select Pred₂:      পিজ্জা (probability: 0.8901)
Keep: পিজ্জা, Discard: Pred₁


═══════════════════════════════════════════
STEP 3: Generate Third Word
═══════════════════════════════════════════

Input to Decoder:  [<START>, আমি, পিজ্জা]
                    ↓
Encoder provides:  [I_vec, like_vec, pizza_vec]
                    ↓
Decoder processes: 3 positions
                    ↓
Output predictions: [Pred₁, Pred₂, Pred₃]
                    ↓
Select Pred₃:      পছন্দ (probability: 0.8756)
Keep: পছন্দ, Discard: Pred₁, Pred₂


═══════════════════════════════════════════
STEP 4: Generate Fourth Word
═══════════════════════════════════════════

Input to Decoder:  [<START>, আমি, পিজ্জা, পছন্দ]
                    ↓
Encoder provides:  [I_vec, like_vec, pizza_vec]
                    ↓
Decoder processes: 4 positions
                    ↓
Output predictions: [Pred₁, Pred₂, Pred₃, Pred₄]
                    ↓
Select Pred₄:      করি (probability: 0.9123)
Keep: করি, Discard: Pred₁, Pred₂, Pred₃


═══════════════════════════════════════════
STEP 5: Generate End Token
═══════════════════════════════════════════

Input to Decoder:  [<START>, আমি, পিজ্জা, পছন্দ, করি]
                    ↓
Encoder provides:  [I_vec, like_vec, pizza_vec]
                    ↓
Decoder processes: 5 positions
                    ↓
Output predictions: [Pred₁, Pred₂, Pred₃, Pred₄, Pred₅]
                    ↓
Select Pred₅:      <END> (probability: 0.8945)
                    ↓
STOP GENERATION!

Final Translation: আমি পিজ্জা পছন্দ করি
```

### Why Discard Earlier Predictions?

**Efficiency**: We only need the prediction for the **newly added token position**.

```
Step 2 produces: [Prediction_for_<START>, Prediction_for_আমি]
                        ↑                          ↑
                   Already have              This is new → Keep!
```

### Masking During Inference - Critical Point!

**Question**: Do we still apply masking during inference when processing sequentially?

**Answer**: YES! Masking is **still applied**.

**Why?**

1. **Architecture Consistency**: Cannot change model architecture between training and inference
2. **Trained Parameters**: Mask layer has learned weights during training
3. **Model Expectation**: Disabling masking would confuse the model

**Example at Step 3**:
```
Input: [<START>, আমি, পিজ্জা]

Mask still applied:
              <START>  আমি  পিজ্জা
<START>    [    0,    -∞,   -∞  ]
আমি        [    0,     0,   -∞  ]
পিজ্জা     [    0,     0,    0  ]

Even though we're processing sequentially,
masking ensures consistency with training behavior.
```

### Complete Inference Example

```
┌─────────────────────────────────────────┐
│ English Input: "I like pizza"          │
│ (Processed once by encoder)             │
└──────────────┬──────────────────────────┘
               │
               ▼
        Encoder Output
     [I_vec, like_vec, pizza_vec]
     (Remains constant throughout)
               │
┌──────────────┴──────────────────────────┐
│                                         │
▼                                         │
Step 1:                                   │
Input: [<START>]                         │
Output: আমি ────┐                        │
                │                         │
Step 2:         │                         │
Input: [<START>, আমি] ◄──────┘           │
Output: পিজ্জা ──────┐                   │
                      │                   │
Step 3:               │                   │
Input: [<START>, আমি, পিজ্জা] ◄──────┘  │
Output: পছন্দ ────────┐                  │
                       │                  │
Step 4:                │                  │
Input: [<START>, আমি, পিজ্জা, পছন্দ] ◄──┘
Output: করি ──────────┐
                       │
Step 5:                │
Input: [<START>, আমি, পিজ্জা, পছন্দ, করি] ◄──┘
Output: <END>
        ↓
   STOP! Translation Complete
   
Result: "আমি পিজ্জা পছন্দ করি"
```

### Inference Strategies

**1. Greedy Decoding** (shown above):
- Always pick highest probability word
- Fast but may miss better overall translations
- Example: Always select argmax(probabilities)

**2. Beam Search**:
- Keep top-k candidates at each step
- Explore multiple paths simultaneously
- Better translation quality

```
Beam Size = 3

Step 1 candidates:
  1. আমি (prob: 0.92)
  2. আমরা (prob: 0.05) - "we"
  3. তুমি (prob: 0.02) - "you"

Step 2 from "আমি":
  1. আমি পিজ্জা (prob: 0.89)
  2. আমি খাবার (prob: 0.04) - "I food"
  3. আমি বার্গার (prob: 0.03) - "I burger"

Continue exploring top beams...
Select path with highest overall probability
```

**3. Sampling**:
- Sample from probability distribution
- More diverse outputs
- Useful for creative tasks

### Training vs Inference: Complete Comparison

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input Source** | Ground truth target | Previously generated tokens |
| **Processing** | Parallel (single pass) | Sequential (autoregressive) |
| **Input Size** | Full sequence at once | Grows incrementally |
| **Masking** | Applied | Applied |
| **Teacher Forcing** | Yes (use correct tokens) | No (use predictions) |
| **Predictions Used** | All positions (for loss) | Only last position kept |
| **Speed** | Fast (parallelized) | Slower (step-by-step) |
| **Encoder Calls** | Once per batch | Once per sentence |
| **Decoder Calls** | Once per sequence | Multiple (one per token) |
| **Stop Condition** | Fixed sequence length | <END> token generated |

---

## Complete Architecture Summary

### Full Decoder Block Diagram

```
┌─────────────────────────────────────────────────────┐
│ INPUT: Bangla Target Sequence                       │
│ <START>, আমি, পিজ্জা, পছন্দ, করি                   │
│ (5 tokens × 512-dimensional embeddings)              │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ + Positional        │
         │   Encoding          │
         │   (Applied once)    │
         └─────────┬───────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             │
┌────────────────────────────┐    │
│ Masked Multi-Head          │    │
│ Attention (8 heads)        │    │
│ • Blocks future tokens     │    │
│ • 5×5 attention matrix     │    │
│ • Causal masking           │    │
└────────────┬───────────────┘    │
             │                    │
             ▼                    │
┌────────────────────────────┐    │
│ Add & Norm                 │    │
│ • Residual: Input + Output │◄───┘
│ • LayerNorm                │
└────────────┬───────────────┘
             │         ┌──────────────────┐
             │         │ Encoder Output   │
             │         │ (English words)  │
             │         │ I, like, pizza   │
             │         └────────┬─────────┘
             │                  │
             ▼                  │
┌────────────────────────────┐  │
│ Cross Attention            │  │
│ • Q from Bangla (decoder) │  │
│ • K,V from English        │◄─┘
│   (encoder)                │
│ • 5×3 attention matrix     │
│ • Dynamic alignment        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Add & Norm                 │
│ • Residual connection      │
│ • LayerNorm                │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Feed-Forward Network       │
│ • Layer 1: 512 → 2048     │
│ • ReLU activation          │
│ • Layer 2: 2048 → 512     │
│ • Introduces non-linearity │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Add & Norm                 │
│ • Final residual           │
│ • Final LayerNorm          │
└────────────┬───────────────┘
             │
             ▼
    [Repeat 6 times]
             │
             ▼
┌────────────────────────────┐
│ Linear + Softmax           │
│ • 512 → 100,000            │
│ • Softmax probabilities    │
│ • Argmax for prediction    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ OUTPUT: Bangla Words       │
│ আমি, পিজ্জা, পছন্দ, করি, <END> │
└────────────────────────────┘
```

### Key Parameters Summary

| Component | Parameters | Details |
|-----------|------------|---------|
| **Input Embedding** | 512 | Word embedding dimension |
| **Positional Encoding** | 512 | Same as embedding dim |
| **Attention Heads** | 8 | Multi-head count |
| **Head Dimension** | 64 | 512 / 8 heads |
| **FFN Hidden** | 2048 | 4× expansion |
| **Decoder Blocks** | 6 | Stacked layers |
| **Vocabulary** | 100,000 | Bangla vocab size |
| **Total Params** | ~65M | Original Transformer |

---

## Key Insights for English-Bangla Translation

### 1. **Word Order Transformation**
```
English SVO: Subject - Verb - Object
Bangla SOV:  Subject - Object - Verb

Cross attention learns this reordering automatically!
```

### 2. **Morphological Richness**
```
Bangla verbs conjugate extensively:
  করি (I do)
  করো (you do - informal)
  করেন (you do - formal)
  করে (he/she does)
  করছি (I am doing)

Decoder learns all patterns through training
```

### 3. **Script Differences**
```
English: Latin alphabet (26 letters)
Bangla: Brahmic script (50+ characters + conjuncts)

Positional encodings adapt to different script characteristics
```

### 4. **Absence of Articles**
```
English: "I like the pizza"
Bangla:  "আমি পিজ্জা পছন্দ করি" (no article)

Cross attention learns to ignore English articles
```

---

## Conclusion

The Transformer decoder elegantly solves sequence generation through:

1. **Masked Multi-Head Attention**: Enables parallel training while preserving causal dependencies. Masking prevents information leakage by blocking future tokens, allowing efficient learning of autoregressive patterns.

2. **Cross Attention**: Creates dynamic bridges between source and target languages. Each target word attends to relevant source words, learning complex mappings like English SVO → Bangla SOV automatically.

3. **Residual Connections & Layer Normalization**: Stabilize training in deep networks (6+ layers), prevent gradient issues, and ensure information flows smoothly.

4. **Feed-Forward Networks**: Introduce essential non-linearity and expand model capacity (4× expansion to 2048 dimensions) for learning complex linguistic patterns.

5. **Training-Inference Duality**: Architecture supports both parallel training (efficient) and sequential inference (necessary for generation), using masking consistently in both modes.

6. **Hierarchical Representations**: Stacked blocks progressively refine understanding from character patterns → words → phrases → sentences → complete semantic meaning.

This architecture revolutionized machine translation, enabling:
- High-quality translations across distant language pairs (English-Bangla)
- Handling of complex morphology and word order differences
- Scalability to massive datasets and model sizes
- Foundation for modern language models (GPT, BERT, T5)

The same principles power multimodal AI, enabling integration of text, images, audio, and other modalities through cross attention mechanisms.

---

## References

- Vaswani et al. (2017). "Attention Is All You Need" - Original Transformer paper
- "The Illustrated Transformer" by Jay Alammar
- "The Annotated Transformer" by Harvard NLP
- Bangla NLP research papers and resources

---
