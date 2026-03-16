# JAX MiniGPT

A minimal **GPT-style language model (~20M parameters)** implemented from scratch using **JAX**.

This project demonstrates the full lifecycle of building a transformer language model: defining the architecture, preprocessing text data, training with modern JAX tooling, checkpointing the model, and generating text through a simple chat interface.

The implementation uses core components from the JAX ecosystem including **Flax/NNX, Grain, Optax, and Orbax** to build a clean and scalable LLM training pipeline.

---

## Features

- GPT-2 style **decoder-only transformer architecture**
- ~20M parameter **MiniGPT language model**
- **JAX-based training pipeline**
- **Flax/NNX neural network modules**
- **Grain data loader for efficient batching**
- **Optax optimizer and learning rate scheduling**
- **Orbax checkpointing**
- **TinyStories dataset preprocessing**
- **Gradio chat interface for text generation**

---

## Architecture

The model follows a standard **decoder-only transformer architecture** similar to GPT-2.

Key components include:

- Token embeddings  
- Positional embeddings  
- Multi-head causal self-attention  
- Transformer blocks  
- Feed-forward layers  
- Residual connections  
- Layer normalization  
- Language modeling head  

Causal masking ensures tokens only attend to previous tokens during generation.

---

## Tech Stack

| Component | Library |
|-----------|--------|
| Numerical Computing | JAX |
| Neural Networks | Flax / NNX |
| Data Pipeline | Grain |
| Optimization | Optax |
| Checkpointing | Orbax |
| Tokenization | tiktoken |
| Interface | Gradio |

---

## Project Structure
jax-minigpt
│
├── 1 Building the LLM Architecture.ipynb
├── 2 Data Loading with Grain.ipynb
├── 3 Training the Model.ipynb
├── 4 Loading and Running a Pre-trained LLM.ipynb
│
├── TinyStories-1000.txt
└── model_checkpoint.orbax


Notebook overview:

- **Building the LLM Architecture** – Implements a GPT-style transformer using JAX and Flax/NNX.
- **Data Loading with Grain** – Preprocesses the TinyStories dataset and constructs the data pipeline.
- **Training the Model** – Defines the training loop, loss function, optimizer, and checkpointing.
- **Loading and Running a Pre-trained LLM** – Loads saved checkpoints and generates text via a chat interface.

---

## Dataset

The model is trained on **TinyStories**, a dataset of short synthetic stories designed for training small language models.

The text is tokenized using the **GPT-2 tokenizer via tiktoken** and converted into fixed-length training sequences.

---

## Training

Training uses **JAX transformations and Optax optimization**.

Key components include:

- Cross-entropy language modeling loss
- JIT-compiled training step
- Gradient updates with Optax
- Learning rate scheduling
- Model checkpointing with Orbax

Example configuration:
max_seq_length = 128
epochs = 3
optimizer = Adam

---

## Inference

After training, the model can generate text by **autoregressively predicting tokens**.

Example prompt:
Once upon a time a little dragon

The model generates a continuation of the story based on the prompt.

---

## Interactive Chat Interface

A **Gradio interface** is included to interact with the trained model.

Users can:

- Enter prompts
- Adjust generation temperature
- Generate story continuations

---

## Learning Goals

This project focuses on understanding the **core mechanics of modern language models**:

- Transformer architectures  
- Autoregressive language modeling  
- JAX primitives (JIT, autodiff, vectorization)  
- Efficient training pipelines for LLMs  

---

## Future Improvements

Possible extensions:

- Distributed training (pmap / pjitted training)
- Larger datasets
- Perplexity evaluation metrics
- KV caching for faster inference
- Flash attention
