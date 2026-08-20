# Transformer Architecture

## 📌 Overview

The **Transformer** is a deep learning architecture introduced in the paper **“Attention Is All You Need” (2017)**.

Unlike traditional RNN/LSTM-based architectures, Transformers do not process sequences strictly one token at a time. Instead, they use **Self-Attention** to understand relationships between tokens in a sequence and can process tokens in parallel during training.

Transformers became the foundation of modern NLP and Generative AI systems such as **BERT, GPT, T5, and many LLMs**.

---

## 🧠 Why Transformers?

### Limitations of RNN/LSTM

RNNs and LSTMs process sequences sequentially:

```text
Token 1 → Token 2 → Token 3 → Token 4
```

This creates several problems:

* Difficult to parallelize
* Long training time
* Difficulty handling very long-range dependencies
* Information can become weaker across long sequences

Transformers solve many of these problems using **Self-Attention**.

```text
Token 1 ─┐
Token 2 ─┼──→ Self-Attention → Contextual Representations
Token 3 ─┤
Token 4 ─┘
```

---

# 🏗️ Transformer Architecture

The original Transformer consists of two major components:

```text
                TRANSFORMER
                    │
          ┌─────────┴─────────┐
          │                   │
       ENCODER             DECODER
          │                   │
   Understand Input       Generate Output
```

### Encoder

The encoder converts the input sequence into contextual representations.

### Decoder

The decoder generates the output sequence using the encoded information.

The original architecture contains:

```text
Input
  ↓
Embedding
  ↓
Positional Encoding
  ↓
Encoder × N
  ↓
Decoder × N
  ↓
Linear Layer
  ↓
Softmax
  ↓
Output
```

---

# 🔤 1. Tokenization

Before entering the Transformer, text is converted into tokens.

Example:

```text
"I love AI"

        ↓

["I", "love", "AI"]

        ↓

[10, 25, 73]
```

The token IDs are then converted into vectors using an **Embedding Layer**.

---

# 🔢 2. Token Embeddings

Each token is represented as a dense vector.

Example:

```text
"AI"

↓

[0.21, -0.13, 0.72, ..., 0.18]
```

If the embedding dimension is `512`, every token gets a vector of size:

```text
512
```

The embedding contains semantic information about the token.

---

# 📍 3. Positional Encoding

Unlike RNNs, Transformers do not inherently know the order of tokens.

For example:

```text
"I love AI"

"I AI love"
```

The Transformer needs information about token positions.

Therefore, **Positional Encoding** is added to the token embeddings.

```text
Token Embedding
       +
Positional Encoding
       ↓
Transformer Input
```

The original Transformer used sinusoidal positional encoding.

Modern Transformer architectures may use other approaches such as:

* Learned positional embeddings
* RoPE (Rotary Positional Embeddings)
* ALiBi

---

# 👀 4. Self-Attention

Self-Attention is the core idea behind the Transformer.

It allows every token to determine which other tokens are important for understanding its meaning.

Example:

```text
"The animal didn't cross the road because it was tired."
```

The model needs to understand what **"it"** refers to.

Self-Attention allows the representation of `"it"` to attend strongly to relevant words such as `"animal"`.

---

# 🔑 Query, Key and Value

Self-Attention uses three representations:

```text
Query (Q)
Key   (K)
Value (V)
```

They are generated from the input representation.

```text
Input X
 │
 ├──→ WQ → Query
 ├──→ WK → Key
 └──→ WV → Value
```

### Query

"What information am I looking for?"

### Key

"What information do I contain?"

### Value

"What information should I pass forward?"

---

# 🧮 Scaled Dot-Product Attention

The attention formula is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

Where:

* `Q` = Query
* `K` = Key
* `V` = Value
* `dₖ` = dimension of the Key vectors

### Step-by-step

```text
QKᵀ
 ↓
Similarity Scores
 ↓
Divide by √dₖ
 ↓
Softmax
 ↓
Attention Weights
 ↓
Multiply with V
 ↓
Contextual Representation
```

---

# 🧠 5. Multi-Head Attention

Instead of performing only one attention operation, Transformers use multiple attention heads.

```text
                 Input
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Head 1       Head 2      Head 3
       ↓           ↓           ↓
   Attention    Attention   Attention
       └───────────┼───────────┘
                   ↓
              Concatenate
                   ↓
              Linear Layer
```

Different heads can learn different relationships.

For example:

* One head may focus on syntax
* Another may focus on subject-object relationships
* Another may focus on long-range dependencies

---

# 🏛️ 6. Transformer Encoder Block

Each encoder layer contains:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed Forward Network
  ↓
Add & Norm
  ↓
Output
```

More explicitly:

```text
                Input
                  │
                  ▼
        Multi-Head Attention
                  │
                  ▼
            Add + LayerNorm
                  │
                  ▼
         Feed Forward Network
                  │
                  ▼
            Add + LayerNorm
                  │
                  ▼
               Output
```

The original Transformer stacks multiple encoder layers.

---

# ⚙️ 7. Feed Forward Neural Network

After attention, each token representation passes through a position-wise feed-forward network.

Typically:

```text
Linear
  ↓
Activation
  ↓
Linear
```

Original Transformer:

```text
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Modern Transformers may use activations such as:

* ReLU
* GELU
* SwiGLU

---

# 🧱 8. Residual Connections

Transformers use residual/skip connections.

Instead of:

```text
X → Layer → Output
```

we have:

```text
X ───────────────┐
│                │
↓                ↓
Layer          Add
│                │
└────────────────┘
```

This helps:

* Gradient flow
* Stable training
* Deep networks

---

# 📏 9. Layer Normalization

Layer Normalization is used around the Transformer sublayers.

The original architecture can be represented as:

```text
X
 ↓
Attention
 ↓
Add
 ↓
LayerNorm
 ↓
Feed Forward
 ↓
Add
 ↓
LayerNorm
```

Modern Transformer implementations may use **Pre-LayerNorm** instead of the original Post-LayerNorm design.

---

# 🏗️ 10. Transformer Decoder

The decoder is slightly different from the encoder.

A decoder block contains:

```text
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention
  ↓
Add & Norm
  ↓
Feed Forward Network
  ↓
Add & Norm
  ↓
Output
```

### Two types of attention

#### 1. Masked Self-Attention

Prevents the decoder from looking at future tokens.

Example:

```text
I love ___
```

While predicting the next token, the model should not see the answer beforehand.

#### 2. Cross-Attention

The decoder attends to the output of the encoder.

```text
Encoder Output
      │
      ↓
Cross-Attention
      ↑
Decoder
```

This is particularly important in sequence-to-sequence tasks such as machine translation.

---

# 🎯 11. Causal / Masked Attention

For autoregressive generation, the model predicts tokens from left to right.

```text
I
I love
I love AI
I love AI very
I love AI very much
```

The attention matrix is masked so a token cannot attend to future tokens.

```text
      I  love  AI  very
I     ✓   ✗    ✗    ✗
love  ✓   ✓    ✗    ✗
AI    ✓   ✓    ✓    ✗
very  ✓   ✓    ✓    ✓
```

This is called **Causal Self-Attention**.

---

# 🔀 Encoder-Decoder vs Decoder-Only

This distinction is extremely important for modern LLMs.

### Encoder-Only

Example:

```text
BERT
```

Used mainly for understanding tasks.

```text
Input → Encoder → Representation
```

Applications:

* Classification
* Sentiment analysis
* Named Entity Recognition
* Embeddings

---

### Decoder-Only

Examples:

```text
GPT
LLaMA
Mistral
```

Used primarily for autoregressive text generation.

```text
Prompt
  ↓
Decoder
  ↓
Next Token
  ↓
Next Token
  ↓
Next Token
```

This is the dominant architecture behind modern generative LLMs.

---

### Encoder-Decoder

Examples:

```text
T5
Original Transformer
```

Useful for sequence-to-sequence tasks.

```text
Input
  ↓
Encoder
  ↓
Context
  ↓
Decoder
  ↓
Output
```

---

# 📊 Transformer Architecture Summary

```text
                 INPUT TEXT
                     │
                     ▼
                Tokenization
                     │
                     ▼
               Token Embedding
                     │
                     +
            Positional Information
                     │
                     ▼
              ┌──────────────┐
              │   ENCODER    │
              │              │
              │ Self-Attn     │
              │ Add & Norm    │
              │ Feed Forward  │
              │ Add & Norm    │
              └──────┬───────┘
                     │
                     ▼
              Encoder Output
                     │
                     ▼
              ┌──────────────┐
              │   DECODER    │
              │              │
              │ Masked Attn   │
              │ Cross-Attn    │
              │ Feed Forward  │
              └──────┬───────┘
                     │
                     ▼
                 Linear
                     │
                     ▼
                  Softmax
                     │
                     ▼
                Output Token
```

---

# 🚀 Why Transformers Changed Deep Learning

Transformers introduced several major advantages:

### Parallelization

RNN:

```text
Token 1 → Token 2 → Token 3 → Token 4
```

Transformer:

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼→ Attention
Token 4 ─┘
```

This makes training much more parallelizable.

### Long-Range Dependencies

Self-Attention provides a direct mechanism for tokens to interact with other tokens, even when they are far apart in the sequence.

### Scalability

Transformers scale effectively with:

* More data
* More parameters
* More compute

This scalability helped enable modern LLMs.

---

# 🤖 Transformers and LLMs

Modern Large Language Models are predominantly based on Transformer architectures.

Examples:

```text
BERT       → Encoder-only
GPT        → Decoder-only
T5         → Encoder-Decoder
LLaMA      → Decoder-only
Mistral    → Decoder-only
```

A simplified GPT-style architecture:

```text
Text
 ↓
Tokenizer
 ↓
Token Embeddings
 ↓
Positional Information
 ↓
Transformer Decoder Blocks
 ↓
Linear Projection
 ↓
Softmax
 ↓
Next Token
```

---

# 🔄 Next Token Prediction

Decoder-only LLMs are generally trained using next-token prediction.

Example:

```text
Input:

"The sky is"

Target:

"blue"
```

The model learns:

```text
P(next token | previous tokens)
```

During generation:

```text
The
 ↓
The sky
 ↓
The sky is
 ↓
The sky is blue
 ↓
The sky is blue today
```

---

# 🧩 Important Concepts to Learn

After understanding the basic Transformer architecture, study these concepts:

* Tokenization
* Embeddings
* Positional Encoding
* Self-Attention
* Query / Key / Value
* Scaled Dot-Product Attention
* Multi-Head Attention
* Causal Attention
* Cross-Attention
* Feed Forward Network
* Residual Connections
* Layer Normalization
* Encoder
* Decoder
* Encoder-only models
* Decoder-only models
* Encoder-Decoder models
* Next Token Prediction

---

# 📚 Learning Progression

A good learning sequence is:

```text
RNN
 ↓
LSTM
 ↓
GRU
 ↓
Seq2Seq
 ↓
Attention Mechanism
 ↓
Transformer
 ↓
BERT / GPT
 ↓
LLMs
 ↓
RAG
 ↓
AI Agents
```

For your **GenAI journey**, the most important transition is:

```text
RNN/LSTM
    ↓
Attention
    ↓
Transformer
    ↓
GPT-style Decoder
    ↓
LLM
    ↓
RAG / Agents
```

---

# 🔑 Key Takeaway

> **The Transformer is fundamentally an attention-based architecture that allows models to capture relationships between tokens efficiently and in parallel.**

The most important equation to remember is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

And the most important conceptual shift is:

```text
RNN/LSTM:
Sequential processing

Transformer:
Attention-based processing
```

---

## 📖 Reference

**Paper:** *Attention Is All You Need*
**Authors:** Vaswani et al.
**Year:** 2017

The Transformer architecture is the foundation of a large part of modern NLP and Generative AI.
