"""
Module 2: Data Shaping & Sampling
==================================

Triển khai 3 chức năng cốt lõi cho Giai đoạn 2 và Giai đoạn 3 của A-MSB-GRPO:

  (A) Augmented Sampling — Mở rộng không gian mẫu từ Layer 1 (N) thành (N*M)
      tại Layer 2 bằng Self-Reflection prompt.
  (B) Semantic Error Profiling — Sử dụng sentence-transformers + K-Means
      để phân cụm lỗi và xác định Top Error Cluster.
  (C) Static Balanced Batch Construction — Lấy mẫu có hoàn lại (Sampling with
      Replacement) để xây dựng Tensor batch tĩnh K=16 (50% Correct / 50% Error).

Quy ước ký hiệu Shape:
  B  = Batch size (số lượng prompt)
  N  = Số rollouts ban đầu per prompt (Layer 1)
  M  = Số nhánh suy luận mới per rollout (Layer 2)
  K  = Kích thước batch tĩnh cuối cùng (mặc định 16)
  D  = Embedding dimension (384 cho all-MiniLM-L6-v2)
  C  = Số cụm K-Means
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_K: int = 16                    # Static batch size
DEFAULT_N_CLUSTERS: int = 4            # Số cụm K-Means
DEFAULT_EMBED_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_M: int = 4                     # Số nhánh self-reflection per rollout

# System prompt trung lập cho Layer 2 Self-Reflection
SELF_REFLECTION_SYSTEM_PROMPT: str = (
    "Hãy kiểm tra lại tính logic của các bước giải. "
    "Giữ nguyên nếu phát hiện đáp án đã chính xác, "
    "hiệu chỉnh lại nếu phát hiện lỗi sai."
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutSample:
    """Một mẫu kết quả từ quá trình sinh (rollout)."""
    prompt: str                     # Truy vấn đầu vào gốc
    response: str                   # Đáp án mô hình sinh ra
    reward: float                   # 0.0 (sai) hoặc 1.0 (đúng)
    token_ids: Optional[torch.Tensor] = None      # (L,)
    logprobs: Optional[torch.Tensor] = None        # (L,)
    ref_logprobs: Optional[torch.Tensor] = None    # (L,)
    attention_mask: Optional[torch.Tensor] = None  # (L,)
    layer: int = 1                  # Layer nguồn (1 hoặc 2)
    parent_index: Optional[int] = None  # Index mẫu cha (nếu Layer 2)


@dataclass
class BatchResult:
    """Kết quả sau khi xây dựng Static Balanced Batch."""
    correct_samples: List[RolloutSample]      # K/2 mẫu đúng
    error_samples: List[RolloutSample]        # K/2 mẫu sai (từ Top Error Cluster)
    cluster_distribution: torch.Tensor        # (C,) phân phối cụm lỗi
    top_cluster_id: int                       # ID cụm lỗi chiếm đa số
    correct_ratio: float                      # Tỷ lệ đúng trước khi balance


# ===========================================================================
# PHẦN A: Evaluate & Pool Splitting (Bước 2.2)
# ===========================================================================

def evaluate_and_split_pools(
    samples: List[RolloutSample],
) -> Tuple[List[RolloutSample], List[RolloutSample], float]:
    """
    Phân tách không gian mẫu thành Correct Pool và Error Pool.

    Args:
        samples: List[RolloutSample] — tập hợp tất cả mẫu (N*M mẫu từ Layer 2,
                 hoặc N mẫu từ Layer 1).

    Returns:
        correct_pool: List[RolloutSample] — các mẫu có reward = 1.0.
        error_pool:   List[RolloutSample] — các mẫu có reward = 0.0.
        correct_ratio: float — tỷ lệ mẫu đúng / tổng mẫu.

    Ghi chú: Hàm này không thay đổi Shape tensor, chỉ phân loại mẫu.
    """
    correct_pool = [s for s in samples if s.reward > 0.5]
    error_pool = [s for s in samples if s.reward <= 0.5]

    total = len(samples)
    correct_ratio = len(correct_pool) / total if total > 0 else 0.0

    return correct_pool, error_pool, correct_ratio


# ===========================================================================
# PHẦN B: Semantic Error Profiling (Bước 2.3)
# ===========================================================================

class SemanticErrorProfiler:
    """
    Phân tích cụm lỗi ngữ nghĩa sử dụng Sentence Embeddings + K-Means.

    Mô hình nhúng chạy trên CPU để giảm tải GPU cho quá trình training.
    Thuật toán K-Means sử dụng scikit-learn.

    Attributes:
        embed_model_name: str — tên mô hình sentence-transformers.
        n_clusters: int — số cụm K-Means.
        _embedder: SentenceTransformer — instance mô hình nhúng (lazy init).
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
        """Lazy-load mô hình nhúng trên CPU."""
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
        Tính embedding vectors cho danh sách văn bản.

        Args:
            texts: List[str] — danh sách các chuỗi đáp án sai.

        Returns:
            embeddings: (N_err, D) — ma trận embedding, D=384 cho MiniLM.

        Ghi chú Shape:
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
        Phân cụm ngữ nghĩa các mẫu lỗi và xác định Top Error Cluster.

        Args:
            error_pool: List[RolloutSample] — tập hợp các mẫu sai.

        Returns:
            labels:       (N_err,)  — nhãn cụm cho mỗi mẫu lỗi.
            embeddings:   (N_err, D) — embedding vectors.
            cluster_dist: (C,)      — phân phối xác suất của C cụm (Tensor, tổng = 1).
            top_cluster:  int       — ID cụm chiếm tỷ trọng lớn nhất.

        Ghi chú Shape:
            error texts      → list, len = N_err
            embeddings       → (N_err, D)
            labels           → (N_err,)
            cluster_counts   → (C,)
            cluster_dist     → (C,)   # normalized to sum=1
        """
        from sklearn.cluster import KMeans

        if len(error_pool) == 0:
            # Không có mẫu lỗi → trả về phân phối đều
            dummy_dist = torch.ones(self.n_clusters) / self.n_clusters  # (C,)
            return np.array([]), np.array([]), dummy_dist, 0

        # Trích xuất văn bản đáp án sai
        error_texts = [s.response for s in error_pool]

        # Tính embeddings (trên CPU)
        embeddings = self.compute_embeddings(error_texts)  # (N_err, D)

        # Điều chỉnh số cụm nếu ít mẫu hơn số cụm
        actual_n_clusters = min(self.n_clusters, len(error_pool))

        # K-Means clustering
        kmeans = KMeans(
            n_clusters=actual_n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings)  # (N_err,)

        # Tính phân phối cụm
        cluster_counts = np.bincount(labels, minlength=actual_n_clusters)  # (C,)
        cluster_counts_float = cluster_counts.astype(np.float64)
        cluster_dist_np = cluster_counts_float / cluster_counts_float.sum()  # (C,)

        # Pad nếu actual_n_clusters < self.n_clusters
        if actual_n_clusters < self.n_clusters:
            padded = np.zeros(self.n_clusters)
            padded[:actual_n_clusters] = cluster_dist_np
            # Re-normalize sau khi pad
            padded = padded / padded.sum()
            cluster_dist_np = padded

        cluster_dist = torch.from_numpy(cluster_dist_np).float()  # (C,)

        # Top Error Cluster = cụm có số lượng mẫu lớn nhất
        top_cluster = int(np.argmax(cluster_counts))

        return labels, embeddings, cluster_dist, top_cluster


# ===========================================================================
# PHẦN C: Static Balanced Batch Construction (Bước 3.1)
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
    Xây dựng Tensor batch tĩnh cân bằng 50/50 bằng Lấy mẫu có hoàn lại.

    Nguyên lý:
      - Lấy K/2 mẫu từ Correct Pool (sampling with replacement nếu thiếu).
      - Lấy K/2 mẫu từ Top Error Cluster (sampling with replacement nếu thiếu).
      - Loại bỏ hoàn toàn dynamic reshaping / padding → ngăn phân mảnh VRAM.

    Args:
        correct_pool:        List[RolloutSample] — tập mẫu đúng.
        error_pool:          List[RolloutSample] — tập mẫu sai (toàn bộ).
        cluster_labels:      (N_err,) — nhãn cụm cho mỗi mẫu trong error_pool.
        top_cluster_id:      int — ID cụm lỗi chiếm đa số.
        cluster_distribution: (C,) — phân phối cụm lỗi.
        K: int               — kích thước batch tĩnh (mặc định 16).
        seed: int or None    — random seed cho reproducibility.

    Returns:
        BatchResult — struct chứa K/2 mẫu đúng, K/2 mẫu sai, và metadata.

    Ghi chú:
        Khi correct_pool hoặc top_error_cluster có ít mẫu hơn K/2,
        thuật toán tự động lấy mẫu có hoàn lại (sampling with replacement)
        để luôn đạt đúng kích thước K/2.
    """
    rng = np.random.RandomState(seed)
    half_k = K // 2

    # ---- Lấy K/2 mẫu đúng ----
    if len(correct_pool) == 0:
        raise ValueError(
            "Correct Pool rỗng! Trường hợp batch 0% đáng lẽ phải được "
            "xử lý bởi NGRPO ở Layer 1 (Conditioning Gate), "
            "không nên vào Layer 2."
        )
    correct_indices = rng.choice(
        len(correct_pool), size=half_k, replace=True  # Sampling with Replacement
    )  # (K/2,)
    sampled_correct = [correct_pool[i] for i in correct_indices]

    # ---- Lấy K/2 mẫu sai từ Top Error Cluster ----
    # Lọc ra các mẫu thuộc cụm lỗi chiếm đa số
    top_error_samples = [
        s for s, label in zip(error_pool, cluster_labels)
        if label == top_cluster_id
    ]

    if len(top_error_samples) == 0:
        # Fallback: nếu top cluster rỗng (edge case), dùng toàn bộ error pool
        top_error_samples = error_pool

    if len(top_error_samples) == 0:
        raise ValueError(
            "Cả Error Pool lẫn Top Error Cluster đều rỗng! "
            "Trường hợp batch 100% đáng lẽ phải được xử lý bởi NGRPO ở Layer 1."
        )

    error_indices = rng.choice(
        len(top_error_samples), size=half_k, replace=True
    )  # (K/2,)
    sampled_errors = [top_error_samples[i] for i in error_indices]

    # Tính correct_ratio gốc (trước balance)
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
    Collate BatchResult thành các Tensor có kích thước tĩnh, sẵn sàng cho forward pass.

    Args:
        batch_result: BatchResult — kết quả từ build_static_balanced_batch().
        max_seq_len: int — chiều dài sequence tối đa (pad/truncate).
        pad_token_id: int — token ID dùng cho padding.
        device: str — thiết bị đích ("cuda" hoặc "cpu").

    Returns:
        Dict chứa các Tensor:
            "input_ids":      (K, max_seq_len) — token IDs đã pad.
            "attention_mask": (K, max_seq_len) — mask (1=valid, 0=pad).
            "old_logprobs":   (K, max_seq_len) — log-probs từ policy cũ.
            "ref_logprobs":   (K, max_seq_len) — log-probs từ reference policy.
            "rewards":        (K,)             — phần thưởng nhị phân.
            "correct_ratio":  ()               — tỷ lệ đúng gốc (scalar).
            "cluster_dist":   (C,)             — phân phối cụm lỗi.

    Ghi chú Shape:
        all_samples    → list, len = K
        token_ids_list → list of (L_i,) tensors
        padded_ids     → (K, max_seq_len)
        padded_mask    → (K, max_seq_len)
        rewards_tensor → (K,)
    """
    all_samples = batch_result.correct_samples + batch_result.error_samples  # len = K

    K = len(all_samples)

    # Khởi tạo Tensor tĩnh đã fill pad
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
