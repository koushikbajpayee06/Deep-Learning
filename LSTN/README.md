# LSTM --- Long Short-Term Memory 🧠

A practical learning repository for understanding **Long Short-Term
Memory (LSTM)** networks --- from the limitations of vanilla RNNs to
LSTM architecture, memory management, gates, mathematical operations,
and next-word prediction using TensorFlow/Keras.

This README is based on the accompanying LSTM study material, which
focuses on both **how LSTM works** and **why it works**.

------------------------------------------------------------------------

## 📌 What is LSTM?

**Long Short-Term Memory (LSTM)** is a specialized type of Recurrent
Neural Network (RNN) designed to handle **long-term dependencies in
sequential data**.

The key idea behind LSTM is a **dual-memory architecture**:

-   **Cell State (`Cₜ`)** → long-term memory
-   **Hidden State (`hₜ`)** → short-term/current context
-   **Gates** → control what information is forgotten, stored, and
    exposed

------------------------------------------------------------------------

## 🎯 Why LSTM?

A standard RNN works well for short sequences, but can struggle when
important information appears many timesteps earlier.

For example:

> "Maharashtra is a beautiful state. It has 25 cities. It has beautiful
> vegetation and forests. Mumbai and ... The language spoken there is
> \_\_\_\_."

The expected answer is **Marathi**, but a vanilla RNN may lose the
earlier "Maharashtra" context over a long sequence.

The study material identifies two major issues:

-   **Vanishing gradients** --- gradients become very small and early
    information can be lost.
-   **Exploding gradients** --- gradients can become very large, causing
    training instability.

LSTM addresses the long-term memory problem through a separate cell
state and controlled information flow.

------------------------------------------------------------------------

## 🔄 RNN → LSTM Evolution

``` text
ANN
 │
 │ Sequential data becomes difficult
 ▼
RNN
 │
 │ Single hidden-state memory
 │ Long-term dependency problem
 ▼
LSTM
 │
 │ Dual memory + gate-controlled information flow
 ▼
Better long-term sequence learning
```

### RNN

``` text
Input → RNN Cell → Hidden State → Output
             ↑
      Previous Hidden State
```

### LSTM

``` text
Previous Cell State ───────────────────────────────► New Cell State
        │
        ▼
   Forget Gate
        │
        ▼
    Input Gate
        │
        ▼
   Cell Update
        │
        ▼
   Output Gate
        │
        ▼
   Hidden State
```

------------------------------------------------------------------------

## 🧩 LSTM Cell Components

At timestep `t`, an LSTM receives **three inputs**:

  -----------------------------------------------------------------------
  Input                   Symbol                  Meaning
  ----------------------- ----------------------- -----------------------
  Previous Cell State     `Cₜ₋₁`                  Long-term memory from
                                                  the previous timestep

  Previous Hidden State   `hₜ₋₁`                  Short-term/current
                                                  context

  Current Input           `xₜ`                    Current word, feature,
                                                  or data point
  -----------------------------------------------------------------------

It produces **two outputs**:

  -----------------------------------------------------------------------
  Output                  Symbol                  Meaning
  ----------------------- ----------------------- -----------------------
  Updated Cell State      `Cₜ`                    Updated long-term
                                                  memory

  Updated Hidden State    `hₜ`                    Updated short-term
                                                  memory/current output
  -----------------------------------------------------------------------

So the LSTM cell can be represented as:

``` text
(Cₜ₋₁, hₜ₋₁, xₜ)
          │
          ▼
      LSTM Cell
          │
      ┌───┴───┐
      ▼       ▼
     Cₜ      hₜ
```

------------------------------------------------------------------------

# 🔐 The Three LSTM Gates

The gates are the main control mechanisms inside an LSTM.

## 1. Forget Gate

### Purpose

The **Forget Gate** decides what information should be discarded from
the previous cell state.

``` text
Previous Memory
      │
      ▼
 Forget Gate
      │
      ▼
Remove outdated information
```

The gate uses the previous hidden state and current input:

``` text
fₜ = sigmoid(Wf · [hₜ₋₁, xₜ] + bf)
```

The sigmoid output is between `0` and `1`:

-   `0` → forget completely
-   `1` → remember completely

------------------------------------------------------------------------

## 2. Input Gate

### Purpose

The **Input Gate** decides what new information should be added to
long-term memory.

``` text
Current Input + Previous Hidden State
                │
                ▼
            Input Gate
                │
                ▼
       Select important information
```

The input gate controls how much new information enters the cell state.

------------------------------------------------------------------------

## 3. Output Gate

### Purpose

The **Output Gate** decides what information should be exposed as the
current hidden state.

``` text
Updated Cell State
       │
       ▼
   Output Gate
       │
       ▼
 Hidden State
```

The hidden state then becomes the short-term memory passed to the next
timestep.

------------------------------------------------------------------------

# 🔄 Complete LSTM Information Flow

The simplified sequence is:

``` text
1. Forget
      ↓
2. Select New Information
      ↓
3. Update Cell State
      ↓
4. Generate Hidden State
```

More formally:

``` text
fₜ = sigmoid(Wf · [hₜ₋₁, xₜ] + bf)

iₜ = sigmoid(Wi · [hₜ₋₁, xₜ] + bi)

C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

oₜ = sigmoid(Wo · [hₜ₋₁, xₜ] + bo)

hₜ = oₜ ⊙ tanh(Cₜ)
```

Where:

-   `sigmoid` → filters information between `0` and `1`
-   `tanh` → produces candidate values between `-1` and `1`
-   `⊙` → element-wise multiplication
-   `Cₜ` → updated long-term memory
-   `hₜ` → updated hidden state

------------------------------------------------------------------------

# 🧠 Cell State vs Hidden State

  -----------------------------------------------------------------------
  Feature                 Cell State              Hidden State
  ----------------------- ----------------------- -----------------------
  Symbol                  `Cₜ`                    `hₜ`

  Role                    Long-term memory        Short-term/current
                                                  representation

  Purpose                 Preserve important      Carry current context
                          historical information  and produce output

  Passed forward          Yes                     Yes
  -----------------------------------------------------------------------

A useful mental model:

``` text
Cell State   → "What should I remember for later?"
Hidden State → "What information is relevant right now?"
```

------------------------------------------------------------------------

# 📊 RNN vs LSTM

  -----------------------------------------------------------------------
  Aspect                  RNN                     LSTM
  ----------------------- ----------------------- -----------------------
  Memory                  Single hidden state     Cell + hidden state

  Information control     Limited                 Gate controlled

  Long-term dependencies  Difficult               Better handling

  Selective memory        No dedicated gates      Forget/Input/Output
                                                  gates

  Architecture            Simpler                 More complex

  Context retention       Poorer for long         Better for long
                          sequences               sequences
  -----------------------------------------------------------------------

The trade-off is straightforward:

> **RNNs are simpler, while LSTMs provide a more sophisticated memory
> mechanism.**

------------------------------------------------------------------------

# 📝 Practical Example: Pronoun Resolution

A practical example from the study material:

``` text
"Ankit is a great girl..."

"Nitish is a YouTuber... he has made several videos..."
```

The LSTM can manage context as the sequence changes:

``` text
Ankit
  ↓
girl
  ↓
Gender context stored

Nitish appears
  ↓
Context switches
  ↓
Old Ankit context becomes less relevant

Nitish → boy
  ↓
"he" becomes the appropriate reference
```

This demonstrates the idea of **selective memory** --- information can
be added, retained, or removed as context changes.

------------------------------------------------------------------------

# 🤖 LSTM as a Next-Word Predictor

The study material also covers a practical next-word prediction
workflow.

### Problem

Given a sequence of words, predict the most likely next word.

Example:

``` text
Input:
"Hi my name is"

Prediction:
"Nitish"
```

### Data Preparation

``` text
Text
 ↓
Tokenization
 ↓
Word → ID
 ↓
Create sequences
 ↓
Padding
 ↓
Training data
```

Example:

``` text
"Hi my name is Nitish"

→ ["Hi"]
→ ["Hi", "my"]
→ ["Hi", "my", "name"]
→ ["Hi", "my", "name", "is"]
```

Training pairs can then be created:

``` text
[Hi]                  → my
[Hi, my]              → name
[Hi, my, name]        → is
[Hi, my, name, is]    → Nitish
```

------------------------------------------------------------------------

# 🏗️ Next-Word Predictor Architecture

The study material uses an architecture similar to:

``` text
Text
  ↓
Tokenizer
  ↓
Word IDs
  ↓
Padding
  ↓
Embedding
  ↓
LSTM
  ↓
Dense
  ↓
Softmax
  ↓
Next Word Probability
```

A demonstrated configuration includes:

-   Vocabulary size: `283`
-   Embedding dimension: `100`
-   LSTM units: `150`
-   Maximum sequence length: `56`
-   Dense output: vocabulary size
-   Activation: `softmax`
-   Loss: categorical crossentropy
-   Optimizer: Adam
-   Learning rate: `0.001`
-   Batch size: `64`
-   Epochs: `100`

------------------------------------------------------------------------

# 💻 TensorFlow / Keras Implementation

A basic LSTM model:

``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(sequence_length, features)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)
```

For a next-word prediction model, the architecture can be:

``` python
model = Sequential()

model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=max_sequence_length
    )
)

model.add(LSTM(150))

model.add(
    Dense(
        vocab_size,
        activation="softmax"
    )
)
```

------------------------------------------------------------------------

# 🧪 Training Workflow

``` text
1. Load text data
        ↓
2. Tokenize text
        ↓
3. Convert words to numbers
        ↓
4. Create input-output sequences
        ↓
5. Pad sequences
        ↓
6. Build LSTM architecture
        ↓
7. Configure hyperparameters
        ↓
8. Train model
        ↓
9. Monitor loss
        ↓
10. Predict next word
```

------------------------------------------------------------------------

# 🌍 Applications

LSTM networks are useful for sequential-data problems such as:

-   Text and NLP
-   Next-word prediction
-   Time-series data
-   Speech/audio processing
-   Sequence classification
-   Context-dependent predictions

The study material also discusses practical next-word prediction use
cases such as mobile keyboards, email composition, code completion, and
chat applications.

------------------------------------------------------------------------

# 🚀 Key Takeaways

### RNN

> Introduced recurrent memory for sequential data.

### Problem

> A single hidden-state memory can struggle with long sequences,
> especially because of vanishing/exploding gradients.

### LSTM

> Introduced a **dual-memory architecture** with controlled information
> flow.

### Three Gates

``` text
Forget Gate → What should I remove?
Input Gate  → What should I store?
Output Gate → What should I expose?
```

### Core Innovation

``` text
Long-term Memory
       +
Short-term Memory
       +
Gate Control
       =
Better Long-Term Sequence Learning
```

------------------------------------------------------------------------

## 📚 Learning Goals

This repository focuses on understanding both:

1.  **HOW LSTM works**
    -   Architecture
    -   Cell state
    -   Hidden state
    -   Gates
    -   Information flow
    -   Mathematical operations
2.  **WHY LSTM works**
    -   RNN limitations
    -   Long-term dependencies
    -   Vanishing gradients
    -   Selective memory
    -   Controlled information flow

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   NLP / Sequential Modeling

------------------------------------------------------------------------

## 📖 Reference

The concepts and examples in this repository are based on the
accompanying LSTM study material covering the evolution from ANN → RNN →
LSTM, LSTM architecture, gate mechanisms, mathematical implementation,
and next-word prediction.

------------------------------------------------------------------------

## ⭐ Final Takeaway

LSTM is more than an `LSTM()` layer in Keras.

The important part is understanding what happens inside it:

``` text
Input
  ↓
Forget old information
  ↓
Select new information
  ↓
Update long-term memory
  ↓
Select relevant output
  ↓
New Hidden State
```

Once the architecture becomes clear, the code becomes much easier to
understand.
