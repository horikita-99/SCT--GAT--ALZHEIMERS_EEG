 🧠 SCT-GAT: EEG-Based Alzheimer's and Dementia Classification Using Graph Attention Networks

This project implements a deep learning framework that combines advanced **time-frequency analysis (Synchrosqueezed Chirplet Transform)** with **Graph Attention Networks (GATs)** to classify Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using EEG data.

The model achieves high classification accuracy and enhances interpretability by modeling inter-channel EEG relationships as graphs and leveraging attention mechanisms.

---

## 📌 Key Features

- ✅ **SCT-Based Feature Extraction**: Captures subtle time–frequency–chirp changes in EEG signals, outperforming traditional spectral methods.
- ✅ **Dynamic Brain Graphs**: EEG signals are converted into graphs where each node represents a channel and edges reflect feature similarity.
- ✅ **Graph Attention Network (GAT)**: Learns to highlight the most relevant inter-channel relationships using attention mechanisms.
- ✅ **Binary and Multi-Class Classification**: Supports both AD vs CN and AD vs FTD vs CN classification tasks.
- ✅ **High Accuracy**:
  - AD vs CN: ~92%
  - AD vs FTD vs CN: ~83%

---

## 🧪 Project Structure

SCT--GAT--ALZHEIMERS_EEG/
├── data/ # EEG data files (e.g., .csv, .mat)
├── preprocessing.py # SCT-based EEG feature extraction
├── graph_builder.py # Dynamic KNN graph construction using cosine similarity
├── model.py # GAT architecture implementation (PyTorch)
├── train.py # Training pipeline (binary & multi-class)
├── utils.py # Metric calculators, visualizations, helpers
├── requirements.txt # Dependencies list
└── README.md # This file


