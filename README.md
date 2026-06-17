# M-SB-GRPO (Adaptive Multi-Layer Semantic-Balanced GRPO)

This project is an advanced Reinforcement Learning (RL) training system for Large Language Models (LLMs), focusing on self-correction capabilities in mathematical reasoning tasks. The project implements the M-SB-GRPO architecture (also known as A-MSB-GRPO), a breakthrough solution that overcomes three major drawbacks of traditional GRPO: Advantage Vanishing, Correction Bias, and Memory Fragmentation.

---

## 1. Core M-SB-GRPO Architecture

The Adaptive Multi-Layer Semantic-Balanced GRPO architecture is a combination of optimization theories from three foundational papers: Multi-Layer GRPO, SEED-GRPO, and the 50/50 mathematical theorem from DIVA-GRPO. The system operates through 3 phases:

### Phase 1: Base Generation and Stabilization
- High-Throughput Rollout: At Layer 1, the model generates N independent sample results in parallel via the vLLM engine.
- NGRPO (Negative-enhanced GRPO): If a batch is extreme (0% correct or 100% correct), the transition to Layer 2 is canceled. The system activates the NGRPO algorithm with a Virtual Reward to calibrate the Advantage, maintaining a continuous loss function space.
- Anti-Correction Bias: If the batch has mixed results, the entire batch is forwarded to Layer 2 to prevent the model from forming a mechanical reflex of simply changing answers.

### Phase 2: Self-Reflection and Profiling
- Augmented Sampling: Each sample at Layer 1 generates M correction branches with a System Prompt requesting logical self-verification.
- Semantic Error Profiling (SEED Theory): Samples with incorrect answers (Error Pool) are embedded into vectors using the `all-MiniLM-L6-v2` model (running on CPU) and clustered using K-Means Clustering.
- The system extracts the Top Error Cluster (the error cluster with the largest proportion) - representing the most dangerous "systematic error" that the model currently believes to be correct.

### Phase 3: Static Tensor and Optimization (Tensor Shaping & Policy Update)
- Static Balanced Batching (DIVA Theory): The system uses "Sampling with Replacement" to shape a training tensor of static size K (e.g., K=16) with a golden ratio of 50/50: 50% from the Correct Pool and 50% from the Top Error Cluster. This maximizes the magnitude of the Gradient Update (according to Theorem C.3 of DIVA-GRPO) and prevents VRAM fragmentation.
- Continuous Loss Scaling: Calculate the Shannon Entropy (H) of the error cluster distribution. The scaling factor Scale = e^(-H) is multiplied directly into the Loss function. When Entropy is low (the model makes systematic errors), the Loss function is maximized to hit the error hard. When Entropy is high (the model guesses randomly), the Loss function is suppressed to prevent weight collapse.

---

## 2. Comparison Baselines

To demonstrate the superiority of the proactive "semantic-balanced batch construction" method in M-SB-GRPO, the project compares directly against 3 Baselines:

1. Vanilla GRPO: A foundational 1-layer architecture (following DeepSeekMath). No self-correction mechanism and no balancing. Used as a starting point.
2. MGRPO (Multi-Layer GRPO): A 2-layer architecture. Layer 2 samples randomly for error correction (no clustering, no balancing). This baseline is used to prove the superiority of "smart batch balancing" over "random error correction".
3. SEED-GRPO: A 1-layer architecture, using Semantic Entropy to passively reduce advantage weights. This baseline is used to compare "proactive learning environment creation" (M-SB-GRPO) versus "passive weight adjustment".

---

## 3. Datasets and Benchmarks

- Training: The combined data file `combined_train.parquet` contains problems and standard solutions.
- Evaluation (Benchmarks): Testing on the most rigorous Math datasets: MATH-500, GSM8K, AIME 2024/2025, and OlympiadBench.
- Metrics: Automatically collect Pass@1, Pass@2, Pass@4, Pass@8 metrics.

---

## 4. Implementation Details (H100 Server Platform)

All pipelines are forced to run under a strict shared configuration suite to ensure absolute fairness in experiments:

- Hardware: 1x NVIDIA H100 GPU (80GB VRAM).
- Base Model: `Qwen/Qwen2.5-7B-Instruct`.
- Resource Optimization:
  - 4-bit QLoRA (`use_peft=True`, `load_in_4bit=True`).
  - Integrated vLLM as the generation engine (Allocating 30% GPU VRAM).
  - Integrated DeepSpeed ZeRO-3 and Flash Attention 2.
- Hyperparameters (Fair Comparison):
  - Epochs: 1
  - Learning Rate: 1.0e-05.
  - Prompt Length = 2048, Completion = 1024.
  - Generative Rollouts: 8 completions/prompt.

---

## 5. Quick Start Guide

The pipeline is 100% automatically orchestrated via Bash scripts.

Run Training for Baselines (Vanilla, MGRPO, SEED):
```bash
cd compare/open-r1
./scripts/run_baselines_h100.sh
```

Aggregate and Visualize Results:
```bash
# Aggregate summary tables
python compare/run_compare_pipeline.py --config compare/pipeline.h100.compare.yaml --stage collect

# Extract charts automatically
python compare/run_compare_pipeline.py --config compare/pipeline.h100.compare.yaml --stage chart --metric pass@1
```
