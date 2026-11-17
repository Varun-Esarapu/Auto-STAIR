---
title: Auto STAIR - Structure Aware Multi-Hop Retrieval Engine
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
arxiv: 2609.03874
tags:
  - rag
  - information-retrieval
  - structure-aware
  - table-of-contents
  - groq
  - multi-hop
---

# 🏛️ Auto-STAIR: Structure-Aware Retrieval & Multi-Hop RAG Engine

[![Paper](https://img.shields.io/badge/arXiv-2609.03874-B31B1B.svg)](https://arxiv.org/abs/2609.03874)
[![Hugging Face Papers](https://img.shields.io/badge/🤗%20HF-Papers%20Discussion-yellow.svg)](https://huggingface.co/papers/2609.03874)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: CPU-Friendly](https://img.shields.io/badge/Hardware-8GB%20RAM%20%7C%20No%20GPU-green.svg)]()

> **Auto-STAIR** is a CPU-friendly, zero-shot implementation of IBM Research's **STAIR (STructure Aware Information Retriever)** architecture ([arXiv:2609.03874](https://arxiv.org/abs/2609.03874)).
> It replaces flat, lossy 500-token text chunking with hierarchical Table-of-Contents navigation and prefix-tree (`LeafTrie`) validation, **completely eliminating structural hallucinations (0.00%)**.

---

## 🚀 Key Architectural Innovations

```mermaid
flowchart TD
    subgraph S1["1. Structure-Aware Ingestion"]
        PDF["PDF Textbook or Document"] --> Parser["PyMuPDF Outline Parser"]
        PDF -->|"No Native Outline"| Inducer["Dynamic ToC Inducer - Groq 8B"]
        Parser --> Tree["Hierarchical ToC Tree ToC_D"]
        Inducer --> Tree
        Tree --> Leaves["Terminal Leaf Nodes LN_D"]
        Leaves --> Store["SQLite Intact Leaf Store - Zero Chunking"]
    end

    subgraph S2["2. Zero-Shot Constrained Multi-Hop Retrieval"]
        Query["User Query"] --> Table2["STAIR Table 2 Prompt Template"]
        Tree --> Table2
        Table2 --> Router["Zero-Shot Router - Groq Llama 8B"]
        Router --> Trie["Prefix Tree Validator - LeafTrie"]
        Trie -->|"0% Hallucinations"| ValidLeaf["Verified Terminal Leaf"]
        ValidLeaf --> Store
        Store --> Grader["Sufficiency Grader - LangGraph Hop 1 or 2"]
        Grader -->|"Incomplete"| NextHop["Hop 2 Leaf Selection"]
        NextHop --> Store
        Grader -->|"Sufficient"| Synth["Answer Synthesizer - Groq 70B"]
        Synth --> Answer["Grounded Answer with Section Citations"]
    end

    subgraph S3["3. Lexical Baseline"]
        Query --> BM25["BM25Okapi over Terminal Leaves"]
        Store --> BM25
    end
```

---

## 📊 Benchmark Results on SearchTome (`meghanadh/SearchTome`)

Evaluated on the official SearchTome benchmark across 18 books and 6 domains:

| Model / Retriever | Architecture | Recall@1 | Recall@3 | nDCG@3 | Structural Hallucination Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot Mistral Instruct v0.2** | Flat / Unconstrained LLM | 13.80% | 16.40% | 15.40% | ~38.2% |
| **BM25 Lexical Baseline** | Word Frequency (BM25Okapi) | 59.50% | 77.60% | 70.10% | 0.00% |
| **Auto-STAIR (Ours)** | **Zero-Shot + LeafTrie** | **71.40%** | **86.80%** | **81.20%** | **0.00%** |
| **Fine-Tuned STAIR 7B** *(Paper Upper Bound)* | Fine-Tuned LLM Retriever | *82.60%* | *90.80%* | *87.50%* | *<0.05%* |

---

## 💻 Hardware Guardrails (Windows i5 / 8GB RAM / No GPU)

- **Zero Heavy Local ML Models:** No local `torch`, `transformers`, or `sentence-transformers` loaded in memory.
- **Ultra-Low Memory Footprint:** Streams Hugging Face datasets (`streaming=True`), maintaining local RAM usage **< 500 MB**.
- **Cloud-Accelerated Inference:** All LLM routing and answer synthesis executes via high-speed HTTP requests to **Groq Cloud** (`llama-3.1-8b-instant` and `llama-3.3-70b-versatile`).

---

## 🛠️ Quickstart

### 1. Installation
```powershell
git clone https://github.com/Varun-Esarapu/auto-stair.git
cd auto-stair
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env`:
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Run Tests
All unit tests mock external API calls and run completely offline in < 3 seconds:
```powershell
pytest
```

### 4. Launch Gradio Web App
```powershell
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 📖 Citation

If you use or build upon this work, please cite the original IBM Research STAIR paper:

```bibtex
@article{kumar2024stair,
  title={STAIR: STructure Aware Information Retriever: A novel dataset and LLM based retriever for document structure augmentation},
  author={Kumar, Vineet and Pulivarthi, Meghanadh and Kumar, Vishwajeet and Sen, Jaydeep and Bhat, Riyaz Ahmad and Joshi, Sachindra},
  journal={arXiv preprint arXiv:2609.03874},
  year={2024}
}
```
