"""
Module 1: Custom A-MSB-GRPO Loss Function
==========================================

Implements the Loss function for the A-MSB-GRPO architecture, integrating:
  (A) NGRPO Loss — Advantage calibration using Virtual Reward for extreme batches (0%/100%).
  (B) Continuous Loss Scaling — Continuously adjusting update magnitude through
      Shannon Entropy of the error cluster distribution (SEED entropy).

All Tensor operations are designed to be differentiable,
avoiding any disruption to the Gradient flow.

Shape notation conventions:
  B  = Batch size (number of prompts in batch)
  N  = Number of generated samples (rollouts) per prompt
  L  = Token sequence length
  V  = Vocabulary size
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------
DEFAULT_VIRTUAL_REWARD: float = -1.0   # Virtual reward for 0% batches
DEFAULT_KL_COEFF: float = 0.01         # KL-divergence penalty coefficient
DEFAULT_CLIP_EPS: float = 0.2          # Clipping threshold for ratio (similar to PPO)
DEFAULT_ENTROPY_SCALE_MIN: float = 0.01  # Minimum floor for entropy scale


# ===========================================================================
# PART A: NGRPO — Negative-enhanced GRPO with Virtual Reward
# ===========================================================================

def compute_per_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log-probability for each token in the output sequence.
    VRAM-optimized version: processes each sample individually (chunked)
    instead of creating a massive log_softmax tensor (B*N, L-1, V) all at once.

    Args:
        logits:    (B*N, L, V)  — output logits from model forward pass.
        input_ids: (B*N, L)     — corresponding token IDs.

    Returns:
        log_probs: (B*N, L)     — log P(token_t | token_{<t}) for each position.
    """
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    device = logits.device

    # Result: (B*N, L), first position always = 0 (no logit for the first token)
    all_logprobs = torch.zeros(batch_size, seq_len, device=device, dtype=logits.dtype)

    # Process each sample to avoid creating a (B*N, L-1, V) tensor at once
    for i in range(batch_size):
        # Shift logits and labels for sample i
        shift_logits_i = logits[i, :-1, :]      # (L-1, V)
        shift_labels_i = input_ids[i, 1:]        # (L-1,)

        # Log-softmax then gather immediately → keep only (L-1,) instead of (L-1, V)
        log_softmax_i = F.log_softmax(shift_logits_i, dim=-1)  # (L-1, V)
        per_token_logp_i = log_softmax_i.gather(
            dim=-1,
            index=shift_labels_i.unsqueeze(-1)   # (L-1, 1)
        ).squeeze(-1)                              # (L-1,)

        # Write to result (position 0 stays = 0, logprobs start from position 1)
        all_logprobs[i, 1:] = per_token_logp_i

        # Free intermediate tensors immediately
        del log_softmax_i, per_token_logp_i

    return all_logprobs


def compute_advantages_with_virtual_reward(
    rewards: torch.Tensor,
    correct_ratio: torch.Tensor,
    virtual_reward: float = DEFAULT_VIRTUAL_REWARD,
) -> torch.Tensor:
    """
    Compute Advantage for each sample, integrating Virtual Reward mechanism (NGRPO).

    Principle:
      - Standard GRPO: A_i = (r_i - mean(r)) / std(r)
      - When batch is homogeneous (all correct or all incorrect), std(r) = 0
        → Advantage = 0 → Gradient vanishes (Advantage Vanishing).
      - NGRPO addresses this by injecting a Virtual Reward into the reward set
        before normalization, producing useful gradients even for extreme batches.

    Conditioning Gate mechanism (continuous, branch-free):
      - gate = correct_ratio * (1 - correct_ratio)
        → gate = 0 when ratio = 0% or 100% (extreme batch → use NGRPO)
        → gate > 0 when 0% < ratio < 100% (mixed batch → standard GRPO)
      - Final Advantage = gate * A_standard + (1 - gate_indicator) * A_ngrpo
        where gate_indicator is a binary mask (but soft-approximated
        via sigmoid to maintain differentiability when needed).

    Args:
        rewards:       (B*N,) — binary rewards {0, 1} from Rule-based Verifier.
        correct_ratio: ()     — scalar, batch correct ratio (0.0 → 1.0).
        virtual_reward: float — virtual reward value injected for NGRPO.

    Returns:
        advantages:    (B*N,) — calibrated advantages, ready to multiply into loss.

    Shape notes:
        rewards             → (B*N,)
        augmented_rewards   → (B*N + 1,)   # appended 1 virtual reward
        mean_aug, std_aug   → ()            # scalar
        A_ngrpo             → (B*N,)        # truncated, virtual element removed
        A_standard          → (B*N,)
        gate                → ()            # scalar
        advantages          → (B*N,)
    """
    bn = rewards.shape[0]  # B*N

    # ----- NGRPO Advantage (always computed, even for mixed batches) -----
    # Append virtual reward to the end of the reward array
    virtual = torch.tensor(
        [virtual_reward], device=rewards.device, dtype=rewards.dtype
    )  # (1,)
    augmented_rewards = torch.cat([rewards, virtual], dim=0)  # (B*N + 1,)

    mean_aug = augmented_rewards.mean()      # ()
    std_aug = augmented_rewards.std() + 1e-8  # () — epsilon to prevent division by zero

    # Normalize and truncate the last virtual element
    A_ngrpo = ((augmented_rewards - mean_aug) / std_aug)[:bn]  # (B*N,)

    # ----- Standard GRPO Advantage -----
    mean_r = rewards.mean()       # ()
    std_r = rewards.std() + 1e-8  # ()
    A_standard = (rewards - mean_r) / std_r  # (B*N,)

    # ----- Conditioning Gate (continuous) -----
    # gate ∈ [0, 0.25], max when ratio = 50%, equals 0 when ratio = 0% or 100%
    gate = correct_ratio * (1.0 - correct_ratio)  # ()

    # Normalize gate to [0, 1]: gate_norm = gate / 0.25 = 4 * gate
    gate_norm = torch.clamp(4.0 * gate, 0.0, 1.0)  # ()

    # Linear blending: when gate_norm ~ 1 (mixed) → use A_standard
    #                  when gate_norm ~ 0 (extreme) → use A_ngrpo
    advantages = gate_norm * A_standard + (1.0 - gate_norm) * A_ngrpo  # (B*N,)

    return advantages


# ===========================================================================
# PART B: Continuous Loss Scaling via Shannon Entropy (SEED)
# ===========================================================================

def compute_seed_entropy_scale(
    cluster_distribution: torch.Tensor,
    scale_min: float = DEFAULT_ENTROPY_SCALE_MIN,
) -> torch.Tensor:
    """
    Compute a continuous Scale factor from Shannon Entropy of the error cluster distribution.

    Principle (SEED — Shannon Entropy-based Error Dampening):
      - High Entropy H → errors are dispersed, random → Low Scale → reduce update magnitude.
      - Low Entropy H → errors are concentrated, systematic → High Scale → amplify gradient.
      - Formula: Scale = exp(-H), with minimum floor = scale_min.

    Args:
        cluster_distribution: (C,) — probability distribution of C error clusters (sum = 1).
                              Example: [0.6, 0.2, 0.1, 0.1] for 4 clusters.
        scale_min: float — minimum floor for the Scale factor, prevents scale → 0.

    Returns:
        scale: () — scalar multiplier for Loss, value ∈ [scale_min, 1.0].

    Shape notes:
        cluster_distribution → (C,)
        p_safe               → (C,)    # clamped to avoid log(0)
        entropy              → ()      # scalar Shannon Entropy
        scale_raw            → ()      # exp(-H)
        scale                → ()      # clamp(scale_raw, min=scale_min)
    """
    # Ensure valid distribution (avoid log(0))
    p_safe = torch.clamp(cluster_distribution, min=1e-10)  # (C,)

    # Shannon Entropy: H = -Σ p_i * log(p_i)
    entropy = -(p_safe * torch.log(p_safe)).sum()  # ()

    # Convert to inversely proportional scale factor
    scale_raw = torch.exp(-entropy)  # ()

    # Clamp to range [scale_min, 1.0]
    scale = torch.clamp(scale_raw, min=scale_min, max=1.0)  # ()

    return scale


# ===========================================================================
# PART C: Combined A-MSB-GRPO Loss Function
# ===========================================================================

def compute_ratio_and_kl(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    clip_eps: float = DEFAULT_CLIP_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute policy ratio (clipped) and KL-divergence penalty.

    Args:
        current_logprobs: (B*N, L) — log-probs from current policy (being trained).
        old_logprobs:     (B*N, L) — log-probs from old policy (at generation time).
        ref_logprobs:     (B*N, L) — log-probs from reference policy (frozen).
        attention_mask:   (B*N, L) — mask for valid tokens (1=valid, 0=pad).
        clip_eps: float   — clipping threshold.

    Returns:
        clipped_ratio: (B*N,) — clipped ratio, averaged over token dimension.
        kl_penalty:    (B*N,) — KL(π_current || π_ref), averaged over tokens.

    Shape notes:
        log_ratio_raw   → (B*N, L)  # log(π_new / π_old) per token
        ratio            → (B*N, L)  # exp(log_ratio_raw)
        clipped          → (B*N, L)
        mask_sum         → (B*N,)    # number of valid tokens per sequence
        clipped_ratio    → (B*N,)    # averaged over tokens
        kl_per_token     → (B*N, L)
        kl_penalty       → (B*N,)
    """
    # Log-ratio between new policy and old policy
    log_ratio_raw = current_logprobs - old_logprobs  # (B*N, L)

    # Importance sampling ratio
    ratio = torch.exp(log_ratio_raw)  # (B*N, L)

    # Clipping for stability (similar to PPO)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)  # (B*N, L)

    # Take min between original ratio and clipped (pessimistic bound)
    ratio_final = torch.min(ratio, clipped)  # (B*N, L)

    # Average over valid tokens
    mask_float = attention_mask.float()          # (B*N, L)
    mask_sum = mask_float.sum(dim=-1).clamp(min=1.0)  # (B*N,)

    clipped_ratio = (ratio_final * mask_float).sum(dim=-1) / mask_sum  # (B*N,)

    # KL divergence: KL(π_current || π_ref) ≈ Σ (ratio_ref - 1 - log(ratio_ref))
    # where ratio_ref = π_current / π_ref
    log_ratio_ref = current_logprobs - ref_logprobs  # (B*N, L)
    ratio_ref = torch.exp(log_ratio_ref)             # (B*N, L)

    # KL approximation: (r - 1 - log(r)), always >= 0
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
    Combined A-MSB-GRPO Loss function.

    Formula:
        L = -Scale * mean( A_i * clipped_ratio_i ) + β * mean( KL_i )

    Where:
        - A_i: Advantage calibrated via NGRPO (Part A).
        - Scale: Factor from Shannon Entropy (Part B). Equals 1.0 at Layer 1.
        - clipped_ratio_i: Clipped importance sampling ratio.
        - β: KL penalty coefficient.

    Args:
        current_logprobs:     (B*N, L) — log P from policy being trained.
        old_logprobs:         (B*N, L) — log P from policy at generation time.
        ref_logprobs:         (B*N, L) — log P from reference policy (frozen).
        rewards:              (B*N,)   — binary rewards {0, 1}.
        attention_mask:       (B*N, L) — valid token mask.
        correct_ratio:        ()       — batch correct ratio [0, 1].
        cluster_distribution: (C,) or None — error cluster distribution (only at Layer 2).
                              None → Default Scale = 1.0 (Layer 1 / NGRPO path).
        virtual_reward: float — virtual reward value for NGRPO.
        kl_coeff: float       — β coefficient for KL penalty.
        clip_eps: float       — clipping threshold.
        entropy_scale_min: float — minimum floor for entropy scale.

    Returns:
        loss: ()         — scalar loss, ready for backward().
        info: dict       — dictionary containing metrics for logging/debugging:
            - "advantages": (B*N,)
            - "entropy_scale": ()
            - "kl_penalty_mean": ()
            - "policy_loss": ()
            - "total_loss": ()
            - "correct_ratio": ()

    Overall shape notes:
        advantages    → (B*N,)
        scale         → ()
        clipped_ratio → (B*N,)
        kl_penalty    → (B*N,)
        policy_loss   → ()        # -Scale * mean(A * ratio)
        kl_loss       → ()        # β * mean(KL)
        total_loss    → ()        # policy_loss + kl_loss
    """
    # ---- Step 1: Compute Advantage (NGRPO / Standard, auto-selected via gate) ----
    advantages = compute_advantages_with_virtual_reward(
        rewards=rewards,
        correct_ratio=correct_ratio,
        virtual_reward=virtual_reward,
    )  # (B*N,)

    # ---- Step 2: Compute Entropy Scale (SEED) ----
    if cluster_distribution is not None:
        scale = compute_seed_entropy_scale(
            cluster_distribution=cluster_distribution,
            scale_min=entropy_scale_min,
        )  # ()
    else:
        # Layer 1 (NGRPO path): no error cluster info → Scale = 1.0
        scale = torch.tensor(1.0, device=rewards.device, dtype=rewards.dtype)  # ()

    # ---- Step 3: Compute Policy Ratio (clipped) and KL Penalty ----
    clipped_ratio, kl_penalty = compute_ratio_and_kl(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        attention_mask=attention_mask,
        clip_eps=clip_eps,
    )  # clipped_ratio: (B*N,), kl_penalty: (B*N,)

    # ---- Step 4: Combine Loss ----
    # Policy loss: maximize advantage-weighted ratio → negate to minimize
    # advantages: (B*N,), clipped_ratio: (B*N,) → element-wise multiply → (B*N,)
    policy_loss = -(scale * (advantages * clipped_ratio).mean())  # ()

    # KL penalty
    kl_loss = kl_coeff * kl_penalty.mean()  # ()

    # Total loss
    total_loss = policy_loss + kl_loss  # ()

    # ---- Step 5: Metrics for logging ----
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
