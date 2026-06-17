# M-SB-GRPO: Adaptive Multi-Layer Semantic-Balanced GRPO

## Overview

This repository implements the Adaptive Multi-Layer Semantic-Balanced Group Relative Policy Optimization (M-SB-GRPO), an advanced reinforcement learning framework tailored for Large Language Models (LLMs). The architecture is specifically designed to enhance the intrinsic self-correction capabilities of LLMs in complex mathematical reasoning tasks. 

M-SB-GRPO addresses three critical bottlenecks observed in conventional GRPO methodologies: Advantage Vanishing, Correction Bias, and Memory Fragmentation. By synthesizing optimization principles from Multi-Layer GRPO, SEED-GRPO, and the theoretical foundations of DIVA-GRPO, this framework establishes a robust, proactive environment for policy optimization.

---

## 1. Methodology and Architecture

The proposed architecture operates sequentially across three distinct phases to ensure optimal policy updates and continuous loss scaling.

### 1.1 Phase I: Base Generation and Stabilization
- **High-Throughput Rollout:** In the initial layer, the model utilizes the vLLM engine to generate $N$ independent trajectories in parallel.
- **Negative-Enhanced GRPO (NGRPO):** To prevent advantage vanishing in homogeneous batches (where empirical accuracy is either 0% or 100%), transition to the subsequent layer is bypassed. The system instead activates the NGRPO algorithm, applying a Virtual Reward mechanism to calibrate the advantage and maintain the continuity of the loss function space.
- **Anti-Correction Bias:** For heterogeneous batches, the entirety of the batch is propagated to the second layer. This comprehensive propagation mitigates the formation of spurious reflexive corrections, ensuring the model relies on systematic logical reasoning.

### 1.2 Phase II: Self-Reflection and Semantic Error Profiling
- **Augmented Sampling:** Trajectories generated in Phase I branch into $M$ verification pathways, guided by a system prompt engineered to elicit explicit logical self-evaluation.
- **Semantic Error Profiling:** Leveraging the theoretical basis of SEED, erroneous samples are mapped into a continuous embedding space utilizing the `all-MiniLM-L6-v2` encoder. These embeddings are subsequently clustered via K-Means.
- **Top Error Extraction:** The system isolates the most prominent error cluster, conceptually representing the most critical "systematic fallacy" that the current policy exhibits high confidence in.

### 1.3 Phase III: Tensor Shaping and Policy Optimization
- **Static Balanced Batching:** Guided by DIVA-GRPO principles, the system employs sampling with replacement to construct a training tensor of static dimensionality $K$ (e.g., $K=16$). The tensor maintains a strict 50/50 equilibrium between the correct outcomes pool and the dominant error cluster. This balanced stochasticity maximizes the gradient update magnitude and mitigates VRAM fragmentation.
- **Continuous Loss Scaling:** The Shannon Entropy ($H$) of the error cluster distribution is computed to dynamically scale the loss function ($Scale = e^{-H}$). In low-entropy states (indicating concentrated systematic errors), the loss magnitude is maximized. In high-entropy states (indicating dispersed, stochastic errors), the loss is attenuated to prevent weight collapse.

---

## 2. Experimental Baselines

To substantiate the efficacy of the semantic-balanced batching paradigm, M-SB-GRPO is benchmarked against three established architectural paradigms:

1. **Vanilla GRPO:** A standard single-layer architecture representative of baseline mathematical reasoning optimization (e.g., DeepSeekMath). It operates without self-correction or batch balancing protocols.
2. **Multi-Layer GRPO (MGRPO):** A two-layer architecture incorporating uniform random sampling for error correction in the second layer. This baseline serves to distinguish the performance gain of semantic clustering from arbitrary multi-layer interventions.
3. **SEED-GRPO:** A single-layer paradigm that integrates Semantic Entropy to passively scale advantage weights, contrasting with the proactive batch-engineering approach of M-SB-GRPO.

---

## 3. Datasets and Evaluation Protocols

- **Training Corpus:** The model is optimized on `combined_train.parquet`, a curated dataset of mathematical propositions and verified deductive solutions.
- **Evaluation Benchmarks:** Generalization and reasoning capacities are evaluated on standardized benchmarks including **MATH-500**, **GSM8K**, **AIME 2024/2025**, and **OlympiadBench**.
- **Metrics:** Model efficacy is quantified through automated Pass@k metrics (specifically Pass@1, Pass@2, Pass@4, and Pass@8).

---

## 4. Implementation Details

To ensure empirical rigor and reproducibility, all comparative experiments are executed under unified environmental and computational constraints:

- **Computational Environment:** 1x NVIDIA H100 GPU (80GB VRAM).
- **Base Architecture:** `Qwen/Qwen2.5-7B-Instruct`.
- **Optimization and Acceleration:**
  - 4-bit Quantized Low-Rank Adaptation (QLoRA) parameter-efficient fine-tuning.
  - vLLM integration for generation acceleration, capped at 30% GPU VRAM allocation.
  - DeepSpeed ZeRO-3 optimization and Flash Attention 2 for memory and computational efficiency.
- **Hyperparameters:**
  - Training Epochs: 1
  - Learning Rate: 1.0e-05
  - Context Window: 2048 prompt tokens, 1024 completion tokens
  - Generative Rollouts: 8 independent responses per prompt

---

## 5. Usage Instructions

### 5.1 Environment Setup

It is highly recommended to isolate the dependencies using Conda or a Python virtual environment.

**Option A: Conda Environment (Recommended)**
```bash
# Create a new conda environment with Python 3.10
conda create -n msb-grpo python=3.10 -y

# Activate the environment
conda activate msb-grpo

# Install dependencies
pip install -r requirements.txt
```

**Option B: Python Virtual Environment (venv)**
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Execution

The experimental pipeline is fully automated through integrated shell scripts.

**Executing the Proposed M-SB-GRPO Architecture:**
```bash
# Execute the primary training and evaluation pipeline for M-SB-GRPO
cd compare/A-MSB-GRPO
bash main.sh
```

**Executing Baseline Training (Vanilla, MGRPO, SEED):**
```bash
# Execute baseline models for empirical comparison
cd compare/open-r1
bash scripts/run_baselines_h100.sh
```

**Aggregating Results and Visualization:**
```bash
# Aggregate evaluation metrics into structured summaries
python compare/run_compare_pipeline.py --config compare/pipeline.h100.compare.yaml --stage collect

# Generate comparative visualizations (e.g., Pass@1 charts)
python compare/run_compare_pipeline.py --config compare/pipeline.h100.compare.yaml --stage chart --metric pass@1
```
