"""
Module 1: Custom A-MSB-GRPO Loss Function
==========================================

Triển khai hàm Loss cho kiến trúc A-MSB-GRPO, tích hợp:
  (A) NGRPO Loss — Hiệu chuẩn Advantage bằng Virtual Reward cho batch cực đoan (0%/100%).
  (B) Continuous Loss Scaling — Điều chỉnh biên độ cập nhật liên tục thông qua
      Shannon Entropy của phân phối cụm lỗi (SEED entropy).

Tất cả phép toán Tensor được thiết kế đảm bảo tính khả vi (differentiable),
không gây gián đoạn luồng Gradient.

Quy ước ký hiệu Shape:
  B  = Batch size (số lượng prompt trong batch)
  N  = Số lượng mẫu sinh ra (rollouts) cho mỗi prompt
  L  = Chiều dài chuỗi token (sequence length)
  V  = Kích thước từ điển (vocabulary size)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Hằng số mặc định (Default Constants)
# ---------------------------------------------------------------------------
DEFAULT_VIRTUAL_REWARD: float = -1.0   # Phần thưởng ảo cho batch 0%
DEFAULT_KL_COEFF: float = 0.01         # Hệ số KL-divergence penalty
DEFAULT_CLIP_EPS: float = 0.2          # Ngưỡng clipping cho ratio (tương tự PPO)
DEFAULT_ENTROPY_SCALE_MIN: float = 0.01  # Sàn tối thiểu cho entropy scale


# ===========================================================================
# PHẦN A: NGRPO — Negative-enhanced GRPO với Virtual Reward
# ===========================================================================

def compute_per_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Tính log-probability của mỗi token trong chuỗi đầu ra.

    Args:
        logits:    (B*N, L, V)  — logits đầu ra từ model forward pass.
        input_ids: (B*N, L)     — token IDs tương ứng.

    Returns:
        log_probs: (B*N, L)     — log P(token_t | token_{<t}) cho mỗi vị trí.

    Ghi chú Shape:
        logits[:, :-1, :]  → (B*N, L-1, V)  # bỏ vị trí cuối (không có target)
        input_ids[:, 1:]   → (B*N, L-1)      # bỏ vị trí đầu (không có logit)
        log_softmax(...)   → (B*N, L-1, V)
        gather(...)        → (B*N, L-1, 1)  → squeeze → (B*N, L-1)
        pad(...)           → (B*N, L)        # pad 0 ở vị trí đầu tiên
    """
    # Shift: logit tại vị trí t dự đoán token tại vị trí t+1
    shift_logits = logits[:, :-1, :].contiguous()   # (B*N, L-1, V)
    shift_labels = input_ids[:, 1:].contiguous()     # (B*N, L-1)

    # Log-softmax trên chiều vocabulary
    log_softmax = F.log_softmax(shift_logits, dim=-1)  # (B*N, L-1, V)

    # Trích xuất log-prob tương ứng với token thực tế
    per_token_logp = log_softmax.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)  # (B*N, L-1, 1)
    ).squeeze(-1)  # (B*N, L-1)

    # Pad vị trí đầu tiên bằng 0 để giữ nguyên shape (B*N, L)
    per_token_logp = F.pad(per_token_logp, (1, 0), value=0.0)  # (B*N, L)

    return per_token_logp


def compute_advantages_with_virtual_reward(
    rewards: torch.Tensor,
    correct_ratio: torch.Tensor,
    virtual_reward: float = DEFAULT_VIRTUAL_REWARD,
) -> torch.Tensor:
    """
    Tính Advantage cho mỗi mẫu, tích hợp cơ chế Virtual Reward (NGRPO).

    Nguyên lý:
      - GRPO chuẩn: A_i = (r_i - mean(r)) / std(r)
      - Khi batch đồng nhất (tất cả đúng hoặc tất cả sai), std(r) = 0
        → Advantage = 0 → Gradient biến mất (Advantage Vanishing).
      - NGRPO khắc phục bằng cách chèn Virtual Reward vào tập thưởng trước
        khi chuẩn hóa, tạo ra gradient hữu ích ngay cả với batch cực đoan.

    Cơ chế Conditioning Gate (liên tục, không rẽ nhánh):
      - gate = correct_ratio * (1 - correct_ratio)
        → gate = 0 khi ratio = 0% hoặc 100% (batch cực đoan → dùng NGRPO)
        → gate > 0 khi 0% < ratio < 100% (batch hỗn hợp → GRPO chuẩn)
      - Advantage cuối = gate * A_standard + (1 - gate_indicator) * A_ngrpo
        Trong đó gate_indicator là binary mask (nhưng được soft-approximate
        bằng sigmoid để giữ tính khả vi trong trường hợp cần thiết).

    Args:
        rewards:       (B*N,) — phần thưởng nhị phân {0, 1} từ Rule-based Verifier.
        correct_ratio: ()     — scalar, tỷ lệ chính xác của batch (0.0 → 1.0).
        virtual_reward: float — giá trị phần thưởng ảo tiêm vào cho NGRPO.

    Returns:
        advantages:    (B*N,) — advantage đã hiệu chuẩn, sẵn sàng để nhân vào loss.

    Ghi chú Shape:
        rewards             → (B*N,)
        augmented_rewards   → (B*N + 1,)   # thêm 1 virtual reward
        mean_aug, std_aug   → ()            # scalar
        A_ngrpo             → (B*N,)        # cắt bỏ phần tử virtual cuối
        A_standard          → (B*N,)
        gate                → ()            # scalar
        advantages          → (B*N,)
    """
    bn = rewards.shape[0]  # B*N

    # ----- NGRPO Advantage (luôn tính, kể cả batch hỗn hợp) -----
    # Chèn phần thưởng ảo vào cuối mảng thưởng
    virtual = torch.tensor(
        [virtual_reward], device=rewards.device, dtype=rewards.dtype
    )  # (1,)
    augmented_rewards = torch.cat([rewards, virtual], dim=0)  # (B*N + 1,)

    mean_aug = augmented_rewards.mean()      # ()
    std_aug = augmented_rewards.std() + 1e-8  # () — epsilon chống chia 0

    # Chuẩn hóa và cắt bỏ phần tử virtual cuối cùng
    A_ngrpo = ((augmented_rewards - mean_aug) / std_aug)[:bn]  # (B*N,)

    # ----- Standard GRPO Advantage -----
    mean_r = rewards.mean()       # ()
    std_r = rewards.std() + 1e-8  # ()
    A_standard = (rewards - mean_r) / std_r  # (B*N,)

    # ----- Conditioning Gate (liên tục) -----
    # gate ∈ [0, 0.25], đạt max khi ratio = 50%, bằng 0 khi ratio = 0% hoặc 100%
    gate = correct_ratio * (1.0 - correct_ratio)  # ()

    # Normalize gate về [0, 1]: gate_norm = gate / 0.25 = 4 * gate
    gate_norm = torch.clamp(4.0 * gate, 0.0, 1.0)  # ()

    # Trộn tuyến tính: khi gate_norm ~ 1 (mixed) → dùng A_standard
    #                   khi gate_norm ~ 0 (extreme) → dùng A_ngrpo
    advantages = gate_norm * A_standard + (1.0 - gate_norm) * A_ngrpo  # (B*N,)

    return advantages


# ===========================================================================
# PHẦN B: Continuous Loss Scaling qua Shannon Entropy (SEED)
# ===========================================================================

def compute_seed_entropy_scale(
    cluster_distribution: torch.Tensor,
    scale_min: float = DEFAULT_ENTROPY_SCALE_MIN,
) -> torch.Tensor:
    """
    Tính hệ số Scale liên tục từ Shannon Entropy của phân phối cụm lỗi.

    Nguyên lý (SEED — Shannon Entropy-based Error Dampening):
      - Entropy H cao → lỗi phân tán, ngẫu nhiên → Scale thấp → giảm biên độ cập nhật.
      - Entropy H thấp → lỗi tập trung, hệ thống → Scale cao → tăng cường gradient.
      - Công thức: Scale = exp(-H), với sàn tối thiểu = scale_min.

    Args:
        cluster_distribution: (C,) — phân phối xác suất của C cụm lỗi (tổng = 1).
                              Ví dụ: [0.6, 0.2, 0.1, 0.1] cho 4 cụm.
        scale_min: float — sàn tối thiểu cho hệ số Scale, ngăn scale → 0.

    Returns:
        scale: () — scalar hệ số nhân vào Loss, giá trị ∈ [scale_min, 1.0].

    Ghi chú Shape:
        cluster_distribution → (C,)
        p_safe               → (C,)    # clamp để tránh log(0)
        entropy              → ()      # scalar Shannon Entropy
        scale_raw            → ()      # exp(-H)
        scale                → ()      # clamp(scale_raw, min=scale_min)
    """
    # Đảm bảo phân phối hợp lệ (tránh log(0))
    p_safe = torch.clamp(cluster_distribution, min=1e-10)  # (C,)

    # Shannon Entropy: H = -Σ p_i * log(p_i)
    entropy = -(p_safe * torch.log(p_safe)).sum()  # ()

    # Chuyển đổi thành hệ số scale tỷ lệ nghịch
    scale_raw = torch.exp(-entropy)  # ()

    # Clamp về khoảng [scale_min, 1.0]
    scale = torch.clamp(scale_raw, min=scale_min, max=1.0)  # ()

    return scale


# ===========================================================================
# PHẦN C: Hàm Loss Tổng hợp A-MSB-GRPO
# ===========================================================================

def compute_ratio_and_kl(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    clip_eps: float = DEFAULT_CLIP_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính policy ratio (clipped) và KL-divergence penalty.

    Args:
        current_logprobs: (B*N, L) — log-probs từ policy hiện tại (đang train).
        old_logprobs:     (B*N, L) — log-probs từ policy cũ (lúc sinh mẫu).
        ref_logprobs:     (B*N, L) — log-probs từ reference policy (frozen).
        attention_mask:   (B*N, L) — mask cho các token hợp lệ (1=valid, 0=pad).
        clip_eps: float   — ngưỡng clipping.

    Returns:
        clipped_ratio: (B*N,) — ratio đã clip, trung bình theo chiều token.
        kl_penalty:    (B*N,) — KL(π_current || π_ref), trung bình theo token.

    Ghi chú Shape:
        log_ratio_raw   → (B*N, L)  # log(π_new / π_old) per token
        ratio            → (B*N, L)  # exp(log_ratio_raw)
        clipped          → (B*N, L)
        mask_sum         → (B*N,)    # số token hợp lệ mỗi sequence
        clipped_ratio    → (B*N,)    # trung bình theo token
        kl_per_token     → (B*N, L)
        kl_penalty       → (B*N,)
    """
    # Log-ratio giữa policy mới và policy cũ
    log_ratio_raw = current_logprobs - old_logprobs  # (B*N, L)

    # Importance sampling ratio
    ratio = torch.exp(log_ratio_raw)  # (B*N, L)

    # Clipping để ổn định (tương tự PPO)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)  # (B*N, L)

    # Lấy min giữa ratio gốc và clipped (pessimistic bound)
    ratio_final = torch.min(ratio, clipped)  # (B*N, L)

    # Trung bình theo token hợp lệ
    mask_float = attention_mask.float()          # (B*N, L)
    mask_sum = mask_float.sum(dim=-1).clamp(min=1.0)  # (B*N,)

    clipped_ratio = (ratio_final * mask_float).sum(dim=-1) / mask_sum  # (B*N,)

    # KL divergence: KL(π_current || π_ref) ≈ Σ (ratio_ref - 1 - log(ratio_ref))
    # Trong đó ratio_ref = π_current / π_ref
    log_ratio_ref = current_logprobs - ref_logprobs  # (B*N, L)
    ratio_ref = torch.exp(log_ratio_ref)             # (B*N, L)

    # Approximation KL: (r - 1 - log(r)), luôn >= 0
    kl_per_token = ratio_ref - 1.0 - log_ratio_ref   # (B*N, L)
    kl_penalty = (kl_per_token * mask_float).sum(dim=-1) / mask_sum  # (B*N,)

    return clipped_ratio, kl_penalty


def amsb_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    correct_ratio: torch.Tensor,
    cluster_distribution: Optional[torch.Tensor] = None,
    virtual_reward: float = DEFAULT_VIRTUAL_REWARD,
    kl_coeff: float = DEFAULT_KL_COEFF,
    clip_eps: float = DEFAULT_CLIP_EPS,
    entropy_scale_min: float = DEFAULT_ENTROPY_SCALE_MIN,
) -> Tuple[torch.Tensor, dict]:
    """
    Hàm Loss tổng hợp A-MSB-GRPO.

    Công thức:
        L = -Scale * mean( A_i * clipped_ratio_i ) + β * mean( KL_i )

    Trong đó:
        - A_i: Advantage đã hiệu chuẩn qua NGRPO (Phần A).
        - Scale: Hệ số từ Shannon Entropy (Phần B). Bằng 1.0 nếu ở Layer 1.
        - clipped_ratio_i: Importance sampling ratio đã clip.
        - β: Hệ số KL penalty.

    Args:
        current_logprobs:     (B*N, L) — log P từ policy đang train.
        old_logprobs:         (B*N, L) — log P từ policy lúc sinh mẫu.
        ref_logprobs:         (B*N, L) — log P từ reference policy (frozen).
        rewards:              (B*N,)   — phần thưởng nhị phân {0, 1}.
        attention_mask:       (B*N, L) — mask token hợp lệ.
        correct_ratio:        ()       — tỷ lệ chính xác của batch [0, 1].
        cluster_distribution: (C,) hoặc None — phân phối cụm lỗi (chỉ có ở Layer 2).
                              None → Scale mặc định = 1.0 (Layer 1 / NGRPO path).
        virtual_reward: float — giá trị phần thưởng ảo cho NGRPO.
        kl_coeff: float       — hệ số β cho KL penalty.
        clip_eps: float       — ngưỡng clipping.
        entropy_scale_min: float — sàn tối thiểu cho entropy scale.

    Returns:
        loss: ()         — scalar loss, sẵn sàng cho backward().
        info: dict       — dictionary chứa các metric để logging/debugging:
            - "advantages": (B*N,)
            - "entropy_scale": ()
            - "kl_penalty_mean": ()
            - "policy_loss": ()
            - "total_loss": ()
            - "correct_ratio": ()

    Ghi chú Shape tổng thể:
        advantages    → (B*N,)
        scale         → ()
        clipped_ratio → (B*N,)
        kl_penalty    → (B*N,)
        policy_loss   → ()        # -Scale * mean(A * ratio)
        kl_loss       → ()        # β * mean(KL)
        total_loss    → ()        # policy_loss + kl_loss
    """
    # ---- Bước 1: Tính Advantage (NGRPO / Standard, tự động chọn qua gate) ----
    advantages = compute_advantages_with_virtual_reward(
        rewards=rewards,
        correct_ratio=correct_ratio,
        virtual_reward=virtual_reward,
    )  # (B*N,)

    # ---- Bước 2: Tính Entropy Scale (SEED) ----
    if cluster_distribution is not None:
        scale = compute_seed_entropy_scale(
            cluster_distribution=cluster_distribution,
            scale_min=entropy_scale_min,
        )  # ()
    else:
        # Layer 1 (NGRPO path): không có thông tin cụm lỗi → Scale = 1.0
        scale = torch.tensor(1.0, device=rewards.device, dtype=rewards.dtype)  # ()

    # ---- Bước 3: Tính Policy Ratio (clipped) và KL Penalty ----
    clipped_ratio, kl_penalty = compute_ratio_and_kl(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        attention_mask=attention_mask,
        clip_eps=clip_eps,
    )  # clipped_ratio: (B*N,), kl_penalty: (B*N,)

    # ---- Bước 4: Tổng hợp Loss ----
    # Policy loss: tối đa hóa advantage-weighted ratio → đặt dấu âm để minimize
    # advantages: (B*N,), clipped_ratio: (B*N,) → element-wise multiply → (B*N,)
    policy_loss = -(scale * (advantages * clipped_ratio).mean())  # ()

    # KL penalty
    kl_loss = kl_coeff * kl_penalty.mean()  # ()

    # Total loss
    total_loss = policy_loss + kl_loss  # ()

    # ---- Bước 5: Metrics cho logging ----
    info = {
        "advantages": advantages.detach(),             # (B*N,)
        "entropy_scale": scale.detach(),                # ()
        "kl_penalty_mean": kl_penalty.mean().detach(),  # ()
        "policy_loss": policy_loss.detach(),             # ()
        "kl_loss": kl_loss.detach(),                     # ()
        "total_loss": total_loss.detach(),               # ()
        "correct_ratio": correct_ratio.detach(),         # ()
    }

    return total_loss, info
