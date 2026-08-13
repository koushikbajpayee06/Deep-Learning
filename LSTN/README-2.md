# Next Word Prediction using LSTM

A simple **Next Word Prediction** project built using **TensorFlow/Keras**.  
The goal of this project is to understand how text is converted into numerical representations and how an **LSTM (Long Short-Term Memory)** network learns sequential patterns to predict the next word.

---

## 🧠 Learning Objective

In this project, I am learning the complete pipeline:

**Raw Text → Tokenization → Input Sequence Generation → Padding → X/y Creation → One-Hot Encoding → Embedding → LSTM → Softmax → Next Word Prediction**

The project is mainly focused on understanding the concepts behind **RNNs, LSTMs, Embeddings, and sequence-based prediction**.

---

# 1. Problem Statement

Given a sequence of words, the model should predict the **next word**.

For example:

```text
Input:
"I am"

Target:
"learning"