# Multi-Head Attention in Transformers: A Concise Guide
Source: https://www.youtube.com/watch?v=LBsyiaEki_8&list=PLuhqtP7jdD8CQTxwVsuiFYGvHtFpNhlR3&index=8

## 1. Introduction

Multi-head attention is the **core mechanism** that enables Transformers to understand language with human-like fluency. While single-head self-attention creates context-aware word representations, it cannot capture the multi-dimensional complexity of natural language.

**Key Problem:** Language requires understanding multiple relationships simultaneously - syntax, semantics, possession, location, time, and more. Single-head attention can only focus on one dominant pattern at a time.

---

## 2. Why Single-Head Attention Falls Short

### Example: Multiple Relationships in One Sentence

**Sentence:** "The keys on the table belong to Sara."

**Relationships to capture:**
- **Spatial:** keys ↔ table (location: "on")
- **Possession:** keys ↔ Sara (ownership: "belong to")
- **Grammatical:** keys ↔ belong (subject-verb)

**Problem:** Single-head attention must prioritize one relationship, diluting others. It cannot effectively capture all three simultaneously.

### The Ambiguity Problem

**Sentence:** "She saw the man with the telescope."

**Two valid interpretations:**
1. She **used** a telescope to see the man (instrument)
2. She saw a man **who had** a telescope (possession)

Single-head attention cannot maintain both interpretations - it must choose one dominant reading.

---

## 3. The Multi-Head Solution

### Core Concept

Instead of one set of weight matrices (W_Q, W_K, W_V), use **H independent sets**:
```
Single-Head:              Multi-Head (H heads):
┌─────────────┐          ┌─────────────┐
│ W_Q, W_K, W_V│          │ W_Q¹, W_K¹, W_V¹│ → Head 1
└─────────────┘          ├─────────────┤
       ↓                  │ W_Q², W_K², W_V²│ → Head 2
One perspective          ├─────────────┤
                         │     ...     │
                         ├─────────────┤
                         │ W_Qᴴ, W_Kᴴ, W_Vᴴ│ → Head H
                         └─────────────┘
                                ↓
                         H perspectives
```

**Each head learns different linguistic patterns independently!**

---

## 4. Architecture and Process

### Step-by-Step Mechanism

**Input:** Sentence "love Apple phones" with embeddings E [3 × 512]

#### Step 1: Create Q, K, V for Each Head
```
For each head h:
  Q_h = E × W_Q^h  [3×512] × [512×64] = [3×64]
  K_h = E × W_K^h  [3×512] × [512×64] = [3×64]
  V_h = E × W_V^h  [3×512] × [512×64] = [3×64]
```

#### Step 2: Compute Attention Per Head
```
head_h = softmax(Q_h × K_h^T / √d_k) × V_h
       = softmax([3×64] × [64×3] / √64) × [3×64]
       = [3×3] × [3×64]
       = [3×64]
```

#### Step 3: Concatenate All Heads
```
Concat(head₁, head₂, ..., head₈)
= [3×64 | 3×64 | ... | 3×64]
= [3×512]
```

#### Step 4: Linear Transformation
```
Output = Concat × W_O
       = [3×512] × [512×512]
       = [3×512]
```

### Complete Formula
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × W_O

where:
  headᵢ = Attention(QW_Qⁱ, KW_Kⁱ, VW_Vⁱ)
  Attention(Q,K,V) = softmax(QK^T/√d_k) × V
```

---

## 5. Why Concatenation and Linear Transformation?

### Purpose of W_O

**Three critical functions:**

1. **Integration:** Combines independent head outputs into unified representation
2. **Weighting:** Learns which heads are important for each context
3. **Filtering:** Removes irrelevant interpretations (e.g., "Apple" as fruit in tech context)

**Analogy:** Orchestra conductor harmonizing individual instruments into coherent music.

---

## 6. Practical Example

**Sentence:** "The keys on the table belong to Sara"
```
HEAD 1 - Spatial Relationships:
  "keys" ←──(0.8)──→ "table"
  "keys" ←──(0.7)──→ "on"
  Captures: WHERE keys are located

HEAD 2 - Possession:
  "keys" ←──(0.9)──→ "Sara"
  "keys" ←──(0.8)──→ "belong"
  Captures: WHO owns the keys

HEAD 3 - Grammatical Structure:
  "keys" ←──(0.9)──→ "belong"
  Captures: Subject-verb relationship

HEAD 4 - Article-Noun:
  "The" ←──(0.95)──→ "keys"
  Captures: Definiteness
```

**After W_O:** Final representation of "keys" contains **all** these perspectives!

---

## 7. Dimensions Analysis

**Standard Transformer (H=8 heads):**
```
Input:  E [3 × 512]
  ↓
Per head: Q, K, V each [3 × 64]
  ↓
Attention output per head: [3 × 64]
  ↓
Concatenate 8 heads: [3 × 512]
  ↓
Linear transform W_O: [512 × 512]
  ↓
Final output: [3 × 512]
```

**Key relationship:** d_k = d_model / H = 512 / 8 = 64

---

## 8. Modern Models: Evolution of Heads
```
Model            Heads    Parameters    Capability
─────────────────────────────────────────────────────
Transformer-Base   8      65M           Good baseline
BERT-Base         12      110M          Strong NLP
GPT-2             12-25   117M-1.5B     Generation
GPT-3             96      175B          Human-level
GPT-4            ~128     ~1.76T        Beyond human
```

**GPT-3 with 96 heads** can interpret each sentence in 96 different ways simultaneously, enabling human-like language understanding!

---

## 9. Head Specialization

Research shows trained heads develop **functional specialization**:
```
Layer 1, Head 5:  Next-word prediction
Layer 2, Head 3:  Syntactic dependencies
Layer 4, Head 7:  Coreference resolution
Layer 8, Head 2:  Semantic integration
Layer 10, Head 11: Discourse boundaries
```

**Important:** These specializations are **learned**, not programmed!

---

## 10. Key Advantages
```
✓ Parallel diverse pattern extraction
✓ Handles ambiguity via multiple interpretations
✓ Captures complex multi-dimensional relationships
✓ Functional specialization through training
✓ Scales with language complexity
```

---

## 11. Practical Guidelines

**Choosing number of heads:**
```
Task Complexity         Recommended Heads
─────────────────────────────────────────
Simple (classification)  4-8 heads
Medium (translation)     8-16 heads
Complex (dialogue)       16-32 heads
SOTA models             32-96+ heads

Constraint: d_model must be divisible by H
Optimal d_k: 32-128
```

---

## 12. Summary

**Multi-head attention = Committee of linguistic experts**

- **Single-head:** One expert, one perspective, limited insight
- **Multi-head:** Multiple experts, diverse perspectives, robust understanding

**The Formula:**
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × W_O
```

**The Result:** AI that processes language through multiple specialized perspectives simultaneously, matching human fluency and comprehension.

**Key Insight:** Just as humans understand language through parallel cognitive processes (syntax, semantics, pragmatics), multi-head attention enables AI to do the same. This is what makes modern models like ChatGPT remarkably capable.

---

## Resources

- **Paper:** "Attention is All You Need" (Vaswani et al., 2017)
- **Visualization:** BertViz for interactive attention exploration
- **Implementation:** Hugging Face Transformers library
