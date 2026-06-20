# LM-Project: Memory-Efficient Autoregressive Language Model Training

## Abstract

This repository provides a comprehensive framework for training autoregressive language models based on the GPT-2 architecture under constrained GPU memory budgets (8 GB VRAM). The system supports both GPT-2 Small (124 M parameters) and GPT-2 Medium (355 M parameters) configurations, and incorporates several memory optimization strategies including 8-bit quantization, gradient checkpointing, mixed-precision training, and gradient accumulation. The training pipeline is designed for the WikiText-103 corpus with full offline operation capability. This document presents the theoretical foundations, architectural design, and implementation details of the training framework.

---

## 1. Theoretical Foundations

### 1.1 Autoregressive Language Modeling

Given a sequence of tokens $\mathbf{x} = (x_1, x_2, \ldots, x_T)$, an autoregressive language model factorizes the joint probability $P(\mathbf{x})$ as a product of conditional probabilities:

$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$

where $x_{<t} = (x_1, \ldots, x_{t-1})$ denotes the prefix of tokens preceding position $t$. The model, parameterized by $\theta$, is trained to minimize the negative log-likelihood (cross-entropy loss) over a training corpus $\mathcal{D}$:

$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^{|\mathbf{x}|} \log P_\theta(x_t \mid x_{<t})$

### 1.2 Causal Self-Attention

The GPT-2 architecture employs multi-head causal self-attention with a lower-triangular mask to enforce the autoregressive property. For an input sequence $X \in \mathbb{R}^{T \times d_{\text{model}}}$, the output of a single attention head is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

where $Q = XW^Q$, $K = XW^K$, $V = XW^V$ are the query, key, and value projections with learned weight matrices $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $d_k$ is the per-head dimension, and $M \in \mathbb{R}^{T \times T}$ is the causal mask defined as:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

The outputs of $h$ attention heads are concatenated and linearly projected:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

### 1.3 Feed-Forward Network

Each transformer block contains a position-wise feed-forward network (FFN) with GELU activation:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and the activation function is the Gaussian Error Linear Unit, approximated as:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

---

## 2. Model Architectures

### 2.1 GPT-2 Small (124 M Parameters)

| Hyperparameter | Value |
|:---|---:|
| Transformer layers ($L$) | 12 |
| Attention heads ($h$) | 12 |
| Model dimension ($d_{\text{model}}$) | 768 |
| FFN inner dimension ($d_{\text{ff}}$) | 3,072 |
| Vocabulary size ($V$) | 50,257 |
| Maximum sequence length | 512 |
| Total parameters | 124,439,808 |

### 2.2 GPT-2 Medium (355 M Parameters)

| Hyperparameter | Value |
|:---|---:|
| Transformer layers ($L$) | 24 |
| Attention heads ($h$) | 16 |
| Model dimension ($d_{\text{model}}$) | 1,024 |
| FFN inner dimension ($d_{\text{ff}}$) | 4,096 |
| Vocabulary size ($V$) | 50,257 |
| Maximum sequence length | 1,024 |
| Total parameters | 354,823,168 |

---

## 3. Memory Optimization Strategies

### 3.1 Gradient Accumulation

To simulate a larger effective batch size $B_{\text{eff}}$ while operating under VRAM constraints, gradients are accumulated over $K$ micro-batches of size $B_{\text{micro}}$:

$$B_{\text{eff}} = B_{\text{micro}} \times K$$

The parameter update rule with gradient accumulation becomes:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}_{\mathcal{B}_k}(\theta_t)$$

where $\mathcal{B}_k$ denotes the $k$-th micro-batch. The default configuration uses $B_{\text{micro}} = 4$ (GPT-2 Small) or $B_{\text{micro}} = 1$ (GPT-2 Medium) with $K = 8$ or $K = 32$, yielding effective batch sizes of 32.

### 3.2 Mixed-Precision Training (FP16)

The framework employs IEEE 754 half-precision floating-point (float16) for forward and backward passes, with a master copy of weights maintained in float32 for numerical stability. The loss scaling factor $\alpha$ prevents underflow in low-magnitude gradients:

$$g_{\text{scaled}} = \alpha \cdot g, \quad \theta_{t+1} = \theta_t - \eta \cdot \frac{g_{\text{scaled}}}{\alpha}$$

### 3.3 Gradient Checkpointing

Rather than storing all intermediate activations for backpropagation, gradient checkpointing recomputes activations on demand during the backward pass. For a network with $L$ layers, this reduces the memory complexity of activations from:

$$O(L \cdot B \cdot T \cdot d_{\text{model}}) \quad \text{to} \quad O(\sqrt{L} \cdot B \cdot T \cdot d_{\text{model}})$$

at the cost of approximately 33% additional forward computation.

### 3.4 8-bit Quantization (BitsAndBytes)

For the GPT-2 Medium variant, the model parameters are quantized to 8-bit precision using the NormalFloat4 (NF4) data type with block-wise quantization. The quantization function $Q: \mathbb{R} \to \{0, 1\}^8$ maps each block of $B_s$ parameters $\mathbf{w} \in \mathbb{R}^{B_s}$ as follows:

$$\tilde{w}_i = Q(w_i; c) = c \cdot \text{round}\left(\frac{\text{clamp}(w_i, -1, 1)}{c}\right), \quad c = \frac{\max(|\mathbf{w}|)}{2^{7} - 1}$$

This reduces the memory footprint of the 355 M parameter model from approximately 1.42 GB (FP32) to roughly 0.36 GB (INT8), enabling training on consumer GPUs with 8 GB VRAM.

### 3.5 Paged 8-bit AdamW Optimizer

The optimizer states (first and second moment estimates) are also stored in 8-bit precision and managed through a unified paging system, further reducing memory consumption. The AdamW update rule with decoupled weight decay is:

$$\begin{aligned} m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\ v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\ \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\ \theta_{t+1} &= \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right) \end{aligned}$$

where $\lambda$ is the weight decay coefficient, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

---

## 4. Learning Rate Schedule

The training employs a cosine annealing schedule with linear warmup. Let $T_{\text{warmup}}$ be the number of warmup steps and $T_{\text{total}}$ the total training steps. The learning rate at step $t$ is:

$$\eta(t) = \begin{cases} \eta_{\text{max}} \cdot \dfrac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\ \eta_{\text{max}} \cdot \dfrac{1}{2}\left(1 + \cos\left(\pi \cdot \dfrac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & \text{if } t \geq T_{\text{warmup}} \end{cases}$$

with $\eta_{\text{max}} = 5 \times 10^{-5}$ and a warmup ratio of 10%.

---

## 5. Evaluation Metrics

### 5.1 Perplexity

The primary evaluation metric is perplexity, defined as the exponentiated average negative log-likelihood per token:

$$\text{PPL}(\mathcal{D}_{\text{val}}) = \exp\left(-\frac{1}{\sum_{\mathbf{x} \in \mathcal{D}_{\text{val}}} |\mathbf{x}|} \sum_{\mathbf{x} \in \mathcal{D}_{\text{val}}} \sum_{t=1}^{|\mathbf{x}|} \log P_\theta(x_t \mid x_{<t})\right)$$

Intuitively, a perplexity of $k$ indicates that the model is as uncertain on average as if it were choosing uniformly among $k$ equally likely candidates at each position.

### 5.2 Text Generation

Text generation employs nucleus sampling (top-$p$) with temperature scaling. Given a prompt $x_{<t}$, the probability of token $x_t = w$ is modified by temperature $T$:

$P_T(w \mid x_{<t}) = \frac{\exp(z_w / T)}{\sum_{w' \in \mathcal{V}} \exp(z_{w'} / T)}$

where $z_w$ denotes the logit for token $w$. Nucleus sampling restricts the sampling space to the smallest set $\mathcal{V}^{(p)} \subset \mathcal{V}$ such that:

$\sum_{w \in \mathcal{V}^{(p)}} P_T(w \mid x_{<t}) \geq p$

The default configuration uses $T = 0.7$ and $p = 0.9$.

---

## 6. Dataset

### 6.1 WikiText-103

The WikiText-103 language modeling dataset consists of verified Good and Featured articles from Wikipedia, comprising over 100 million tokens. The training split contains approximately 28,000 articles with 103,219,124 tokens, while the validation split contains 60 articles with 217,646 tokens. The dataset preserves original casing, punctuation, and numerical expressions, making it more representative of natural text than preprocessed alternatives.

### 6.2 Preprocessing

Text samples are tokenized using the GPT-2 byte-pair encoding (BPE) tokenizer with a vocabulary of $V = 50{,}257$ tokens. Sequences are either truncated or padded to a fixed maximum length $L_{\max}$:

$$\tilde{\mathbf{x}} = \begin{cases} (x_1, \ldots, x_{L_{\max}}) & \text{if } |\mathbf{x}| \geq L_{\max} \\ (x_1, \ldots, x_{|\mathbf{x}|}, \underbrace{\texttt{[EOS]}, \ldots, \texttt{[EOS]}}_{L_{\max} - |\mathbf{x}|}) & \text{if } |\mathbf{x}| < L_{\max} \end{cases}$$

---

## 7. Project Structure

```
lm-project/
  config.py                     Centralized hyperparameter configuration
  train_gpt.py                  GPT-2 Small (124M) training entry point
  modified_train_gpt.py         GPT-2 Medium (355M) training with 8-bit quantization
  offline_gpt2_medium_train.py  Fully offline GPT-2 Medium training with weight transfer
  transfer_weights.py           Weight initialization transfer (Small to Medium)
  evaluate_model.py             Perplexity evaluation and generation quality assessment
  generate_text.py              Interactive text generation interface
  download_gpt2_medium.py       Pre-download script for offline operation
  verify_env.py                 Runtime environment verification
  verify_dataset.py             WikiText dataset availability verification
  run_training.sh               Shell launcher for conda-based execution
  monitor_training.sh           GPU utilization and training progress monitor
  data/                         Cached datasets and tokenized sequences
  models/                       Saved model checkpoints and final weights
  results/                      Training outputs and intermediate checkpoints
  logs/                         Training log files
```

---

## 8. Hardware Requirements and Configuration

The training pipeline is optimized for an NVIDIA GeForce RTX 4060 (8 GB VRAM). The following table summarizes the memory optimization configuration for each model variant:

| Component | GPT-2 Small | GPT-2 Medium |
|:---|---:|---:|
| Model parameters | 124 M | 355 M |
| Micro-batch size | 4 | 1 |
| Gradient accumulation steps | 8 | 32 |
| Effective batch size | 32 | 32 |
| FP16 mixed precision | Enabled | Enabled |
| Gradient checkpointing | Enabled | Enabled |
| 8-bit quantization | Disabled | Enabled (NF4) |
| Optimizer | AdamW (fused) | Paged AdamW 8-bit |
| Approximate VRAM usage | 6.2 GB | 7.5 GB |

---

## 9. Installation and Execution

### 9.1 Environment Setup

A conda environment with Python 3.10 or later is required. The core dependencies are PyTorch 2.0+, Hugging Face Transformers 4.40.0+, Datasets, and BitsAndBytes.

### 9.2 Dataset Preparation

```bash
python verify_dataset.py
```

### 9.3 Training

For GPT-2 Small:

```bash
python train_gpt.py
```

For GPT-2 Medium (offline mode with weight transfer):

```bash
python offline_gpt2_medium_train.py
```

### 9.4 Evaluation

```bash
python evaluate_model.py
```

### 9.5 Text Generation

```bash
python generate_text.py
```

---

## 10. License and Citation

This project is provided for research and educational purposes. If you use this codebase in your work, please cite accordingly.

---

## References

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Technical Report*.

2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 5998-6008.

3. Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. In *Advances in Neural Information Processing Systems (NeurIPS)*.

4. Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*.

5. Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016). Training deep nets with sublinear memory cost. *arXiv preprint arXiv:1604.06174*.

6. Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2017). Pointer sentinel mixture models. In *International Conference on Learning Representations (ICLR)*.

7. Hendrycks, D. and Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv preprint arXiv:1606.08415*.

8. Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). The curious case of neural text degeneration. In *International Conference on Learning Representations (ICLR)*.
