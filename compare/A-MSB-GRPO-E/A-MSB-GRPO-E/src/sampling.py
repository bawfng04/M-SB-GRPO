"""
Module 2: Data Shaping & Sampling
==================================

Implements 3 core functionalities for Phase 2 and Phase 3 of A-MSB-GRPO:

  (A) Augmented Sampling — Expands the sample space from Layer 1 (N) to (N*M)
      at Layer 2 using Self-Reflection prompts.
  (B) Semantic Error Profiling — Uses sentence-transformers + K-Means
      to cluster errors and identify the Top Error Cluster.
  (C) Static Balanced Batch Construction — Uses Sampling with Replacement
      to build a static Tensor batch of size K=16 (50% Correct / 50% Error).

Shape notation conventions:
  B  = Batch size (number of prompts in batch)
  N  = Number of initial rollouts per prompt (Layer 1)
  M  = Number of new reasoning branches per rollout (Layer 2)
  K  = Final static batch size (default 16)
  D  = Embedding dimension (384 for all-MiniLM-L6-v2)
  C  = Number of K-Means clusters
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------
DEFAULT_K: int = 16                    # Static batch size
DEFAULT_N_CLUSTERS: int = 4            # Number of K-Means clusters
DEFAULT_EMBED_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_M: int = 4                     # Number of self-reflection branches per rollout

# Neutral system prompt for Layer 2 Self-Reflection
SELF_REFLECTION_SYSTEM_PROMPT: str = (
    "Review the logical consistency of each solution step. "
    "Keep the answer unchanged if it is already correct, "
    "and revise it if any errors are found."
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutSample:
    """A single sample result from a rollout generation."""
    prompt: str                     # Original input query
    response: str                   # Model-generated answer
    reward: float                   # 0.0 (incorrect) or 1.0 (correct)
    token_ids: Optional[torch.Tensor] = None      # (L,)
    logprobs: Optional[torch.Tensor] = None        # (L,)
    ref_logprobs: Optional[torch.Tensor] = None    # (L,)
    attention_mask: Optional[torch.Tensor] = None  # (L,)
    layer: int = 1                  # Source layer (1 or 2)
    parent_index: Optional[int] = None  # Parent sample index (if Layer 2)


@dataclass
class BatchResult:
    """Result after building a Static Balanced Batch."""
    correct_samples: List[RolloutSample]      # K/2 correct samples
    error_samples: List[RolloutSample]        # K/2 error samples (from Top Error Cluster)
    cluster_distribution: torch.Tensor        # (C,) error cluster distribution
    top_cluster_id: int                       # ID of the dominant error cluster
    correct_ratio: float                      # Correct ratio before balancing


# ===========================================================================
# PART A: Evaluate & Pool Splitting (Step 2.2)
# ===========================================================================

def evaluate_and_split_pools(
    samples: List[RolloutSample],
) -> Tuple[List[RolloutSample], List[RolloutSample], float]:
    """
    Split the sample space into Correct Pool and Error Pool.

    Args:
        samples: List[RolloutSample] — all samples (N*M samples from Layer 2,
                 or N samples from Layer 1).

    Returns:
        correct_pool: List[RolloutSample] — samples with reward = 1.0.
        error_pool:   List[RolloutSample] — samples with reward = 0.0.
        correct_ratio: float — ratio of correct samples / total samples.

    Note: This function does not change tensor shapes, only classifies samples.
    """
    correct_pool = [s for s in samples if s.reward > 0.5]
    error_pool = [s for s in samples if s.reward <= 0.5]

    total = len(samples)
    correct_ratio = len(correct_pool) / total if total > 0 else 0.0

    return correct_pool, error_pool, correct_ratio


# ===========================================================================
# PART B: Semantic Error Profiling (Step 2.3)
# ===========================================================================

class SemanticErrorProfiler:
    """
    Semantic error clustering analysis using Sentence Embeddings + K-Means.

    The embedding model runs on CPU to offload GPU for the training process.
    The K-Means algorithm uses scikit-learn.

    Attributes:
        embed_model_name: str — sentence-transformers model name.
        n_clusters: int — number of K-Means clusters.
        _embedder: SentenceTransformer — embedding model instance (lazy init).
    """

    def __init__(
        self,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        n_clusters: int = DEFAULT_N_CLUSTERS,
    ):
        self.embed_model_name = embed_model_name
        self.n_clusters = n_clusters
        self._embedder = None  # Lazy initialization

    def _get_embedder(self):
        """Lazy-load the embedding model on CPU."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.embed_model_name, device="cpu"
            )
        return self._embedder

    def compute_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Compute embedding vectors for a list of texts.

        Args:
            texts: List[str] — list of incorrect answer strings.

        Returns:
            embeddings: (N_err, D) — embedding matrix, D=384 for MiniLM.

        Shape notes:
            texts (list, len=N_err) → embeddings (N_err, 384)
        """
        embedder = self._get_embedder()
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )  # (N_err, D)
        return embeddings

    def cluster_errors(
        self,
        error_pool: List[RolloutSample],
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, int]:
        """
        Semantically cluster error samples and identify the Top Error Cluster.

        Args:
            error_pool: List[RolloutSample] — the set of incorrect samples.

        Returns:
            labels:       (N_err,)  — cluster label for each error sample.
            embeddings:   (N_err, D) — embedding vectors.
            cluster_dist: (C,)      — probability distribution of C clusters (Tensor, sum = 1).
            top_cluster:  int       — ID of the cluster with the largest proportion.

        Shape notes:
            error texts      → list, len = N_err
            embeddings       → (N_err, D)
            labels           → (N_err,)
            cluster_counts   → (C,)
            cluster_dist     → (C,)   # normalized to sum=1
        """
        from sklearn.cluster import KMeans

        if len(error_pool) == 0:
            # No error samples → return uniform distribution
            dummy_dist = torch.ones(self.n_clusters) / self.n_clusters  # (C,)
            return np.array([]), np.array([]), dummy_dist, 0

        # Extract incorrect answer texts
        error_texts = [s.response for s in error_pool]

        # Compute embeddings (on CPU)
        embeddings = self.compute_embeddings(error_texts)  # (N_err, D)

        # Adjust number of clusters if fewer samples than clusters
        actual_n_clusters = min(self.n_clusters, len(error_pool))

        # K-Means clustering
        kmeans = KMeans(
            n_clusters=actual_n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings)  # (N_err,)

        # Compute cluster distribution
        cluster_counts = np.bincount(labels, minlength=actual_n_clusters)  # (C,)
        cluster_counts_float = cluster_counts.astype(np.float64)
        cluster_dist_np = cluster_counts_float / cluster_counts_float.sum()  # (C,)

        # Pad if actual_n_clusters < self.n_clusters
        if actual_n_clusters < self.n_clusters:
            padded = np.zeros(self.n_clusters)
            padded[:actual_n_clusters] = cluster_dist_np
            # Re-normalize after padding
            padded = padded / padded.sum()
            cluster_dist_np = padded

        cluster_dist = torch.from_numpy(cluster_dist_np).float()  # (C,)

        # Top Error Cluster = cluster with the most samples
        top_cluster = int(np.argmax(cluster_counts))

        return labels, embeddings, cluster_dist, top_cluster


# ===========================================================================
# PART C: Static Balanced Batch Construction (Step 3.1)
# ===========================================================================

def build_static_balanced_batch(
    correct_pool: List[RolloutSample],
    error_pool: List[RolloutSample],
    cluster_labels: np.ndarray,
    top_cluster_id: int,
    cluster_distribution: torch.Tensor,
    K: int = DEFAULT_K,
    seed: Optional[int] = None,
) -> BatchResult:
    """
    Build a 50/50 statically balanced Tensor batch using Sampling with Replacement.

    Principle:
      - Sample K/2 items from Correct Pool (with replacement if insufficient).
      - Sample K/2 items from Top Error Cluster (with replacement if insufficient).
      - Completely eliminates dynamic reshaping / padding → prevents VRAM fragmentation.

    Args:
        correct_pool:        List[RolloutSample] — correct samples.
        error_pool:          List[RolloutSample] — error samples (all).
        cluster_labels:      (N_err,) — cluster label for each sample in error_pool.
        top_cluster_id:      int — ID of the dominant error cluster.
        cluster_distribution: (C,) — error cluster distribution.
        K: int               — static batch size (default 16).
        seed: int or None    — random seed for reproducibility.

    Returns:
        BatchResult — struct containing K/2 correct samples, K/2 error samples, and metadata.

    Note:
        When correct_pool or top_error_cluster has fewer samples than K/2,
        the algorithm automatically uses sampling with replacement
        to always achieve exactly K/2 size.
    """
    rng = np.random.RandomState(seed)
    half_k = K // 2

    # ---- Sample K/2 correct samples ----
    if len(correct_pool) == 0:
        raise ValueError(
            "Correct Pool is empty! The 0% batch case should have been "
            "handled by NGRPO at Layer 1 (Conditioning Gate), "
            "and should not reach Layer 2."
        )
    correct_indices = rng.choice(
        len(correct_pool), size=half_k, replace=True  # Sampling with Replacement
    )  # (K/2,)
    sampled_correct = [correct_pool[i] for i in correct_indices]

    # ---- Sample K/2 error samples from Top Error Cluster ----
    # Filter samples belonging to the dominant error cluster
    top_error_samples = [
        s for s, label in zip(error_pool, cluster_labels)
        if label == top_cluster_id
    ]

    if len(top_error_samples) == 0:
        # Fallback: if top cluster is empty (edge case), use the entire error pool
        top_error_samples = error_pool

    if len(top_error_samples) == 0:
        raise ValueError(
            "Both Error Pool and Top Error Cluster are empty! "
            "The 100% batch case should have been handled by NGRPO at Layer 1."
        )

    error_indices = rng.choice(
        len(top_error_samples), size=half_k, replace=True
    )  # (K/2,)
    sampled_errors = [top_error_samples[i] for i in error_indices]

    # Compute original correct_ratio (before balancing)
    total_original = len(correct_pool) + len(error_pool)
    original_correct_ratio = len(correct_pool) / total_original if total_original > 0 else 0.5

    return BatchResult(
        correct_samples=sampled_correct,
        error_samples=sampled_errors,
        cluster_distribution=cluster_distribution,
        top_cluster_id=top_cluster_id,
        correct_ratio=original_correct_ratio,
    )


def collate_batch_to_tensors(
    batch_result: BatchResult,
    max_seq_len: int,
    pad_token_id: int = 0,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Collate BatchResult into statically-sized Tensors, ready for forward pass.

    Args:
        batch_result: BatchResult — result from build_static_balanced_batch().
        max_seq_len: int — maximum sequence length (pad/truncate).
        pad_token_id: int — token ID used for padding.
        device: str — target device ("cuda" or "cpu").

    Returns:
        Dict containing Tensors:
            "input_ids":      (K, max_seq_len) — padded token IDs.
            "attention_mask": (K, max_seq_len) — mask (1=valid, 0=pad).
            "old_logprobs":   (K, max_seq_len) — log-probs from old policy.
            "ref_logprobs":   (K, max_seq_len) — log-probs from reference policy.
            "rewards":        (K,)             — binary rewards.
            "correct_ratio":  ()               — original correct ratio (scalar).
            "cluster_dist":   (C,)             — error cluster distribution.

    Shape notes:
        all_samples    → list, len = K
        token_ids_list → list of (L_i,) tensors
        padded_ids     → (K, max_seq_len)
        padded_mask    → (K, max_seq_len)
        rewards_tensor → (K,)
    """
    all_samples = batch_result.correct_samples + batch_result.error_samples  # len = K

    K = len(all_samples)

    # Initialize statically-filled pad Tensors
    input_ids = torch.full((K, max_seq_len), pad_token_id, dtype=torch.long)   # (K, max_seq_len)
    attention_mask = torch.zeros((K, max_seq_len), dtype=torch.long)           # (K, max_seq_len)
    old_logprobs = torch.zeros((K, max_seq_len), dtype=torch.float32)          # (K, max_seq_len)
    ref_logprobs = torch.zeros((K, max_seq_len), dtype=torch.float32)          # (K, max_seq_len)
    rewards = torch.zeros(K, dtype=torch.float32)                              # (K,)

    for i, sample in enumerate(all_samples):
        if sample.token_ids is not None:
            seq_len = min(sample.token_ids.shape[0], max_seq_len)
            input_ids[i, :seq_len] = sample.token_ids[:seq_len]
            attention_mask[i, :seq_len] = 1

        if sample.logprobs is not None:
            seq_len = min(sample.logprobs.shape[0], max_seq_len)
            old_logprobs[i, :seq_len] = sample.logprobs[:seq_len]

        if sample.ref_logprobs is not None:
            seq_len = min(sample.ref_logprobs.shape[0], max_seq_len)
            ref_logprobs[i, :seq_len] = sample.ref_logprobs[:seq_len]

        if sample.attention_mask is not None:
            seq_len = min(sample.attention_mask.shape[0], max_seq_len)
            attention_mask[i, :seq_len] = sample.attention_mask[:seq_len]

        rewards[i] = sample.reward

    return {
        "input_ids": input_ids.to(device),                # (K, max_seq_len)
        "attention_mask": attention_mask.to(device),       # (K, max_seq_len)
        "old_logprobs": old_logprobs.to(device),           # (K, max_seq_len)
        "ref_logprobs": ref_logprobs.to(device),           # (K, max_seq_len)
        "rewards": rewards.to(device),                     # (K,)
        "correct_ratio": torch.tensor(
            batch_result.correct_ratio, device=device
        ),                                                 # ()
        "cluster_dist": batch_result.cluster_distribution.to(device),  # (C,)
    }
