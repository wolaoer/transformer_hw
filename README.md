# transformer_hw
Here’s a concise and clear **README.md** for your repository:

---

# Transformer Model Research Report

This project implements a simplified **Transformer Decoder** model using **PyTorch**, based on the architecture proposed in *“Attention Is All You Need”* (Vaswani et al., 2017). The implementation includes training, evaluation, ablation studies, and text generation experiments on the **WikiText-2** dataset.

---

## 📘 Overview

* Implements **multi-head self-attention**, **positional encoding**, and **feed-forward networks** from scratch.
* Conducts **ablation experiments** (no attention / no positional encoding).
* Supports **text generation** and **perplexity evaluation**.
* Uses **ModelScope** for dataset loading and **Qwen tokenizer** for text processing.

---

## 🧩 Repository Structure

```
project_root/
│── dataset/                # Raw and preprocessed datasets
│── src/                 # Training and evaluation scripts
│   │── model.py         # Transformer model implementation
│   │── train.py         # Training loop
│   │── generate.py          # Evaluation and generation
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/wolaoer/transformer_hw.git
cd transformer_hw
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model

```bash
python transformer_hw.py
```

### Generate text samples

```bash
python transformer_hw.py --generate
```

---

## 🧠 Experiments

* **Dataset:** WikiText-2 (via ModelScope)
* **Optimizer:** Adam, LR = 1e-3
* **Batch size:** 16
* **Epochs:** 10
* **Model:** 2 layers, 4 heads, embedding dim = 128, FFN dim = 512

---

## 📊 Results (Example)

| Model Variant          | Val Loss | Perplexity |
| ---------------------- | -------- | ---------- |
| Baseline (Full)        | 5.21     | 184.08     |
| No Positional Encoding | ↑        | ↑          |
| No Attention           | ↑↑       | ↑↑         |

---

## 🧰 Environment

* **Python:** 3.10
* **PyTorch:** 2.6.0 + CUDA 12.4
* **GPU:** NVIDIA A100 (80GB)

---

## 📎 Reference

Vaswani et al., *“Attention Is All You Need”*, NeurIPS 2017.
[GitHub Repository](https://github.com/wolaoer/transformer_hw)

---
