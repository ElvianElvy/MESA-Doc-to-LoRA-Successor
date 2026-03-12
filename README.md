# MESA: Macular Ephemeral State-space Architecture
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/arXiv-Paper_Draft-b31b1b.svg)](https://arxiv.org/)

**MESA** is an experimental AI architecture designed as the scalable successor to recent context-to-weight compilation models like **Doc-to-LoRA**. It achieves **$O(1)$ KV-Cache Space Complexity** by dynamically compiling text into neural weights in a single forward pass, entirely eliminating the need to store context in VRAM activations.

By introducing **Dual-Objective Generative Regularization (KL-Divergence)**, MESA successfully solves the Stability-Plasticity dilemma inherent in zero-shot weight injection, allowing for high factual plasticity while preserving mathematically flawless base-model grammar.

### 🌟 Key Features
- **$O(1)$ VRAM Scaling:** Maintains a flat memory footprint ($\sim1.05$ GB on Qwen-0.5B) regardless of document context length.
- **Linear-Time Routing:** System-1 Macular Skimming parses 1.5M tokens in `<0.02s` to route salient data to the Hypernetwork.
- **Grammar Stability:** KL-Divergence meta-learning loss successfully drives $D_{KL}$ to $0.0588$, protecting the LLM's linguistic manifolds during injection.

---

## ⚙️ Installation

```bash
git clone https://github.com/ElvianElvy/MESA Project.git
cd MESA-Doc-to-LoRA-Successor
pip install torch transformers datasets accelerate tqdm