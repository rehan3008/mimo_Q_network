# Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning for Accelerated Offline RL

PyTorch implementation of  
**Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning**  
[[Paper](https://ieeexplore.ieee.org/document/10537203)]  <!-- Replace with actual arXiv link if available -->

---

## 📘 Overview

Offline reinforcement learning (RL) eliminates the need for interactive data collection, making it a safe and scalable alternative for real-world applications. However, it suffers from **extrapolation errors** caused by **out-of-distribution (OOD)** data in static datasets.

This paper proposes an Uncertainty-Aware Rank-One MIMO Q Network Framework, a novel framework that addresses these challenges through:

- ✅ **Uncertainty quantification** over OOD samples using a single  network
- 📈 **Lower confidence bound optimization** to train robust policies
- 💡 **Rank-One Multi-Input Multi-Output (MIMO)** architecture that mimics ensemble behavior with minimal computational cost

 This framework significantly improves sample efficiency, robustness, and speed while outperforming prior methods on the **D4RL benchmark**.

---

## 🚀 Features

- 🔍 **Uncertainty-aware training objective** based on lower confidence bound of Q-values
- 🧠 Lightweight **Rank-One MIMO Q-Network** for modeling aleatoric and epistemic uncertainty
- 🧪 Plug-and-play architecture compatible with standard offline RL pipelines (e.g., CQL, TD3+BC)
- 📊 State-of-the-art performance on D4RL tasks with high computational efficiency

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/uar-mimo-offline-rl.git
cd uar-mimo-offline-rl
pip install -r requirements.txt
```

## 📚 Citation
If you use this codebase or refer to the method in your research, please cite:

```bibtex
@article{nguyen2024uncertainty,
  title={Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning},
  author={Nguyen, Thanh and Luu, Tung M and Ton, Tri and Kim, Sungwoong and Yoo, Chang D},
  journal={IEEE Access},
  volume={12},
  pages={100972--100982},
  year={2024},
  publisher={IEEE}
}
```
