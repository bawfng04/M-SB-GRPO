"""
Module 3: Generation & Training Loop
======================================

Vòng lặp huấn luyện chính cho kiến trúc A-MSB-GRPO, xử lý:

  (A) Dynamic Routing — Conditioning Gate phân luồng batch giữa
      NGRPO (Layer 1 only) và Full Pipeline (Layer 1 → Layer 2).
  (B) vLLM Generation Engine — sinh mẫu tốc độ cao với vLLM,
      quản lý VRAM để tránh xung đột với PyTorch training engine.
  (C) PyTorch Training Engine — forward pass + backward pass qua
      LoRA adapter + DeepSpeed ZeRO optimizer.

Quy ước:
  - vLLM và PyTorch KHÔNG chạy đồng thời trên cùng GPU.
  - Sau mỗi bước sinh (generate), vLLM cache được giải phóng trước
    khi PyTorch bắt đầu forward/backward.

Phần cứng mục tiêu: 1x NVIDIA H100 80GB.
"""

import os
import gc
import json
import logging
import math
import random
import time
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from tqdm import tqdm

from src.amsb_loss import (
    amsb_grpo_loss,
    compute_per_token_logprobs,
    DEFAULT_KL_COEFF,
    DEFAULT_CLIP_EPS,
    DEFAULT_VIRTUAL_REWARD,
    DEFAULT_ENTROPY_SCALE_MIN,
)
from src.sampling import (
    RolloutSample,
    BatchResult,
    SemanticErrorProfiler,
    evaluate_and_split_pools,
    build_static_balanced_batch,
    collate_batch_to_tensors,
    SELF_REFLECTION_SYSTEM_PROMPT,
    DEFAULT_K,
    DEFAULT_M,
    DEFAULT_N_CLUSTERS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Hằng số mặc định cho Training
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS: int = 2048
DEFAULT_MAX_SEQ_LEN: int = 4096
DEFAULT_N_ROLLOUTS: int = 8            # N: số rollouts Layer 1
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_LEARNING_RATE: float = 1e-5
DEFAULT_NUM_EPOCHS: int = 3
DEFAULT_LORA_R: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 4

# Pool of diverse self-reflection instructions for Layer 2
SELF_REFLECTION_INSTRUCTIONS: List[str] = [
    "Please review the above solution and provide the final answer.",
    "Carefully check each step for logical or computational errors, then give the corrected final answer.",
    "Re-examine the reasoning chain above. If the solution is correct, confirm it; otherwise, fix any mistakes and provide the correct answer.",
    "Identify any flaws in the previous solution. Provide a revised and accurate final answer.",
    "Verify whether the approach and calculations are correct. Output the final answer after your review.",
    "Critically analyze the solution step by step. Correct any errors you find and present the final answer.",
    "Double-check the mathematical reasoning above. If you spot an error, rework the problem and give the right answer.",
    "Evaluate the correctness of the previous attempt. Provide the final answer, making corrections if necessary.",
    "Look for any mistakes in logic or arithmetic in the solution above, then state the correct final answer.",
    "Reassess the solution from scratch. Confirm the answer if it is correct, or derive the correct one if it is not.",
]


# ===========================================================================
# PHẦN A: VRAM Management Utilities
# ===========================================================================

def flush_vram():
    """
    Giải phóng toàn bộ VRAM cache không sử dụng.
    Gọi sau khi vLLM hoàn tất sinh mẫu, trước khi PyTorch bắt đầu training.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info(f"VRAM flushed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def log_vram_usage(tag: str = ""):
    """Log trạng thái bộ nhớ GPU hiện tại."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"[VRAM {tag}] Allocated: {alloc:.2f} GB | Cached: {cached:.2f} GB")


# ===========================================================================
# PHẦN B: Rule-based Verifier (MATH benchmark)
# ===========================================================================

def verify_math_answer(predicted: str, ground_truth: str) -> float:
    """
    Trình đánh giá dựa trên quy tắc cho MATH benchmark.

    So khớp đáp án cuối cùng (thường nằm trong \\boxed{}).
    Trả về 1.0 nếu đúng, 0.0 nếu sai.

    Args:
        predicted: str — đáp án mô hình sinh ra.
        ground_truth: str — đáp án đúng.

    Returns:
        float — 1.0 hoặc 0.0.
    """
    import re

    def extract_boxed(text: str) -> str:
        """Trích xuất nội dung trong \\boxed{...}."""
        # Tìm \boxed{...} pattern cuối cùng
        matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
        if matches:
            return matches[-1].strip()
        return text.strip()

    pred_answer = extract_boxed(predicted)
    true_answer = extract_boxed(ground_truth)

    # Chuẩn hóa: bỏ khoảng trắng, lowercase
    pred_norm = pred_answer.replace(" ", "").lower()
    true_norm = true_answer.replace(" ", "").lower()

    return 1.0 if pred_norm == true_norm else 0.0


# ===========================================================================
# PHẦN C: vLLM Generation Engine
# ===========================================================================

class VLLMGenerationEngine:
    """
    Engine sinh mẫu tốc độ cao bằng vLLM.

    Quản lý vòng đời vLLM instance:
      - Khởi tạo (init) → sinh mẫu (generate) → giải phóng (destroy).
      - Sau khi gọi destroy(), VRAM được giải phóng hoàn toàn cho PyTorch.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        gpu_memory_utilization: float = 0.45,
        tensor_parallel_size: int = 1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _init_engine(self):
        """Khởi tạo vLLM LLM instance (lazy)."""
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        logger.info(f"Initializing vLLM engine: {self.model_name}")
        self._llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_seq_len,
            dtype="auto",
            enforce_eager=True,  # Tắt CUDA graph để tiết kiệm VRAM
        )
        log_vram_usage("after vLLM init")

    def generate(
        self,
        prompts: List[str],
        n: int = DEFAULT_N_ROLLOUTS,
        system_prompt: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Sinh n mẫu đáp án cho mỗi prompt.

        Args:
            prompts: List[str] — danh sách truy vấn (len = B).
            n: int — số mẫu sinh per prompt.
            system_prompt: Optional[str] — system prompt (cho Layer 2).

        Returns:
            List[List[str]] — len = B, mỗi phần tử là list n đáp án.
        """
        from vllm import SamplingParams

        self._init_engine()

        sampling_params = SamplingParams(
            n=n,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Format prompts nếu có system prompt
        if system_prompt:
            formatted_prompts = [
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                for p in prompts
            ]
        else:
            formatted_prompts = [
                f"<|im_start|>user\n{p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                for p in prompts
            ]

        outputs = self._llm.generate(formatted_prompts, sampling_params)

        results = []
        for output in outputs:
            responses = [o.text for o in output.outputs]
            results.append(responses)

        return results

    def destroy(self):
        """Giải phóng vLLM engine và VRAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            flush_vram()
            logger.info("vLLM engine destroyed & VRAM flushed.")


# ===========================================================================
# PHẦN D: PyTorch Training Engine (LoRA + DeepSpeed)
# ===========================================================================

class TrainingEngine:
    """
    Engine huấn luyện PyTorch với LoRA adapter và DeepSpeed optimizer.

    Workflow:
      1. Load base model (quantized nếu QLoRA).
      2. Attach LoRA adapter (PEFT).
      3. Wrap với DeepSpeed.
      4. Forward pass → tính log-probs & loss.
      5. Backward pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        deepspeed_config: Optional[Dict] = None,
        use_qlora: bool = True,
        micro_batch_size: int = 2,
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.deepspeed_config = deepspeed_config
        self.use_qlora = use_qlora
        self.micro_batch_size = micro_batch_size

        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self._ds_engine = None

    def init_model(self):
        """
        Khởi tạo model (LoRA/QLoRA), reference model, và tokenizer.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Load Base Model (QLoRA nếu bật) ----
        if self.use_qlora:
            logger.info("Loading model with 4-bit QLoRA quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            logger.info("Loading model in bfloat16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        # ---- Attach LoRA Adapter ----
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # ---- Reference Model (frozen, dùng cho KL penalty) ----
        # Dùng chung base model nhưng disable LoRA adapter để lấy logits gốc
        # Tiết kiệm VRAM: không load thêm model mới
        logger.info("Reference model: using base model with LoRA disabled for KL computation.")

        # ---- Bật Gradient Checkpointing để tiết kiệm VRAM ----
        self.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

        log_vram_usage("after training model init")

    def init_deepspeed(self):
        """Wrap model với DeepSpeed engine."""
        import deepspeed

        ds_config = self.deepspeed_config or self._default_ds_config()

        self._ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )
        self.model = self._ds_engine.module
        logger.info("DeepSpeed engine initialized.")
        log_vram_usage("after DeepSpeed init")

    def init_standalone(self):
        """
        Khởi tạo optimizer thường (không DeepSpeed).
        Phù hợp cho QLoRA trên 1 GPU vì model 4-bit chỉ chiếm ~5GB.
        """
        # Chỉ optimize LoRA parameters (trainable)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        logger.info(f"Standalone AdamW optimizer initialized with {len(trainable_params)} param groups.")
        log_vram_usage("after standalone init")

    def _default_ds_config(self) -> Dict:
        """Cấu hình DeepSpeed ZeRO-3 mặc định."""
        return {
            "train_batch_size": DEFAULT_K,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": {"enabled": False},
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": 100,
                    "total_num_steps": 10000,
                },
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 50,
        }

    def compute_logprobs_for_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_ref: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass để tính per-token log-probs.
        Sử dụng micro-batch để tận dụng GPU parallelism trong khi kiểm soát VRAM.

        Args:
            input_ids:      (K, L) — token IDs.
            attention_mask: (K, L) — attention mask.
            use_ref: bool — nếu True, disable LoRA adapter để tính ref logprobs.

        Returns:
            logprobs: (K, L) — per-token log probabilities.
        """
        if use_ref:
            self.model.disable_adapter_layers()

        batch_size = input_ids.shape[0]
        # Inference (no_grad) có thể dùng micro-batch lớn hơn vì không cần lưu computation graph
        mbs = batch_size if use_ref else self.micro_batch_size
        all_logprobs = []

        context_manager = torch.no_grad() if use_ref else torch.enable_grad()
        with context_manager:
            for start in range(0, batch_size, mbs):
                end = min(start + mbs, batch_size)
                ids_mb = input_ids[start:end]           # (mbs, L)
                mask_mb = attention_mask[start:end]      # (mbs, L)

                outputs = self.model(
                    input_ids=ids_mb,
                    attention_mask=mask_mb,
                )
                logits_mb = outputs.logits               # (mbs, L, V)

                logprobs_mb = compute_per_token_logprobs(logits_mb, ids_mb)  # (mbs, L)
                all_logprobs.append(logprobs_mb.detach())

                del outputs, logits_mb, logprobs_mb

                # Smart cache clearing: chỉ flush khi VRAM > 70% capacity
                mem_allocated = torch.cuda.memory_allocated()
                mem_total = torch.cuda.get_device_properties(0).total_memory
                if mem_allocated / mem_total > 0.70:
                    torch.cuda.empty_cache()

        logprobs = torch.cat(all_logprobs, dim=0)  # (K, L)

        if use_ref:
            self.model.enable_adapter_layers()

        return logprobs

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        kl_coeff: float = DEFAULT_KL_COEFF,
        clip_eps: float = DEFAULT_CLIP_EPS,
        virtual_reward: float = DEFAULT_VIRTUAL_REWARD,
        entropy_scale_min: float = DEFAULT_ENTROPY_SCALE_MIN,
    ) -> Dict[str, float]:
        """
        Một bước huấn luyện hoàn chỉnh với gradient accumulation per-sample.

        Để tránh OOM, mỗi mẫu được forward → loss → backward riêng biệt,
        chỉ giữ 1 computation graph tại mỗi thời điểm.

        Args:
            batch: Dict — output từ collate_batch_to_tensors().
            kl_coeff, clip_eps, virtual_reward, entropy_scale_min: hyperparams.

        Returns:
            Dict[str, float] — metrics cho logging.
        """
        input_ids = batch["input_ids"]            # (K, L)
        attention_mask = batch["attention_mask"]   # (K, L)
        old_logprobs = batch["old_logprobs"]       # (K, L)
        rewards = batch["rewards"]                 # (K,)
        correct_ratio = batch["correct_ratio"]     # ()
        cluster_dist = batch["cluster_dist"]       # (C,)

        batch_size = input_ids.shape[0]

        # ---- Bước 1: Tính ref_logprobs (không gradient, rẻ) ----
        ref_logprobs = self.compute_logprobs_for_batch(
            input_ids, attention_mask, use_ref=True
        )  # (K, L) — detached, no grad

        # ---- Bước 2: Tính advantages và entropy scale (chỉ dùng rewards) ----
        from src.amsb_loss import (
            compute_advantages_with_virtual_reward,
            compute_seed_entropy_scale,
        )

        advantages = compute_advantages_with_virtual_reward(
            rewards=rewards,
            correct_ratio=correct_ratio,
            virtual_reward=virtual_reward,
        )  # (K,)

        if cluster_dist is not None and cluster_dist.sum() > 0:
            scale = compute_seed_entropy_scale(
                cluster_distribution=cluster_dist,
                scale_min=entropy_scale_min,
            )
        else:
            scale = torch.tensor(1.0, device=rewards.device, dtype=rewards.dtype)

        # ---- Bước 3: Micro-batch gradient accumulation ----
        # Forward micro-batch → loss per sample → backward → free
        # micro_batch_size > 1 tận dụng GPU parallelism bên trong matrix ops
        if self.optimizer:
            self.optimizer.zero_grad()

        total_loss_val = 0.0
        total_kl_val = 0.0
        mbs = self.micro_batch_size

        for start in range(0, batch_size, mbs):
            end = min(start + mbs, batch_size)
            mb_size = end - start

            ids_mb = input_ids[start:end]            # (mbs, L)
            mask_mb = attention_mask[start:end]       # (mbs, L)

            # Forward micro-batch với gradient
            outputs_mb = self.model(
                input_ids=ids_mb,
                attention_mask=mask_mb,
            )
            logits_mb = outputs_mb.logits              # (mbs, L, V)

            # Tính logprobs cho micro-batch
            cur_logprobs_mb = compute_per_token_logprobs(logits_mb, ids_mb)  # (mbs, L)

            # Tính ratio và KL cho micro-batch
            log_ratio_raw = cur_logprobs_mb - old_logprobs[start:end]
            ratio = torch.exp(log_ratio_raw)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            ratio_final = torch.min(ratio, clipped)

            mask_float = mask_mb.float()
            mask_sum = mask_float.sum(dim=-1).clamp(min=1.0)  # (mbs,)
            clipped_ratio = (ratio_final * mask_float).sum(dim=-1) / mask_sum  # (mbs,)

            # KL per sample trong micro-batch
            log_ratio_ref = cur_logprobs_mb - ref_logprobs[start:end]
            ratio_ref = torch.exp(log_ratio_ref)
            kl_per_token = ratio_ref - 1.0 - log_ratio_ref
            kl = (kl_per_token * mask_float).sum(dim=-1) / mask_sum  # (mbs,)

            # Loss cho micro-batch (sum over samples, scale by total batch_size)
            adv_mb = advantages[start:end]  # (mbs,)
            policy_loss = -(scale * adv_mb * clipped_ratio)  # (mbs,)
            kl_loss = kl_coeff * kl  # (mbs,)
            mb_loss = (policy_loss + kl_loss).sum() / batch_size

            # Backward → giải phóng computation graph
            mb_loss.backward()

            total_loss_val += (policy_loss.sum().item() + kl_loss.sum().item())
            total_kl_val += kl.sum().item()

            del outputs_mb, logits_mb, cur_logprobs_mb, mb_loss

            # Smart cache clearing
            mem_allocated = torch.cuda.memory_allocated()
            mem_total = torch.cuda.get_device_properties(0).total_memory
            if mem_allocated / mem_total > 0.70:
                torch.cuda.empty_cache()

        # ---- Bước 6: Optimizer step ----
        if self._ds_engine is not None:
            self._ds_engine.step()
        elif self.optimizer:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # ---- Metrics ----
        metrics = {
            "total_loss": total_loss_val / batch_size,
            "policy_loss": (total_loss_val - total_kl_val * kl_coeff) / batch_size,
            "kl_loss": total_kl_val * kl_coeff / batch_size,
            "entropy_scale": scale.item(),
            "kl_penalty_mean": total_kl_val / batch_size,
            "correct_ratio": correct_ratio.item() if isinstance(correct_ratio, torch.Tensor) else correct_ratio,
            "mean_advantage": advantages.mean().item(),
        }

        return metrics


# ===========================================================================
# PHẦN E: Main Training Loop A-MSB-GRPO
# ===========================================================================

class AMSBGRPOTrainer:
    """
    Orchestrator chính cho pipeline A-MSB-GRPO.

    Quản lý toàn bộ workflow:
      Layer 1: vLLM generate → evaluate → conditioning gate
        → NGRPO (extreme) hoặc Layer 2 (mixed)
      Layer 2: vLLM generate (self-reflection) → semantic error profiling
        → static balanced batch → train step
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        n_rollouts: int = DEFAULT_N_ROLLOUTS,
        m_reflections: int = DEFAULT_M,
        batch_size_k: int = DEFAULT_K,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        kl_coeff: float = DEFAULT_KL_COEFF,
        clip_eps: float = DEFAULT_CLIP_EPS,
        virtual_reward: float = DEFAULT_VIRTUAL_REWARD,
        entropy_scale_min: float = DEFAULT_ENTROPY_SCALE_MIN,
        vllm_gpu_util: float = 0.45,
        use_qlora: bool = True,
        deepspeed_config: Optional[Dict] = None,
        output_dir: str = "./checkpoints",
    ):
        self.model_name = model_name
        self.n_rollouts = n_rollouts
        self.m_reflections = m_reflections
        self.batch_size_k = batch_size_k
        self.n_clusters = n_clusters
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.virtual_reward = virtual_reward
        self.entropy_scale_min = entropy_scale_min
        self.output_dir = output_dir

        # Sub-engines (lazy init)
        self.vllm_engine = VLLMGenerationEngine(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            gpu_memory_utilization=vllm_gpu_util,
        )
        self.training_engine = TrainingEngine(
            model_name=model_name,
            learning_rate=learning_rate,
            deepspeed_config=deepspeed_config,
            use_qlora=use_qlora,
        )
        self.error_profiler = SemanticErrorProfiler(
            n_clusters=n_clusters,
        )

        # Tracking
        self.global_step = 0
        self.metrics_log: List[Dict] = []

        # Resume state (populated by _load_checkpoint)
        self._resume_epoch = 0
        self._resume_prompt_idx = 0

    def _create_rollout_samples(
        self,
        prompt: str,
        responses: List[str],
        ground_truth: str,
        layer: int = 1,
    ) -> List[RolloutSample]:
        """Tạo RolloutSample từ output vLLM, với error handling cho verify."""
        samples = []
        for resp in responses:
            try:
                reward = verify_math_answer(resp, ground_truth)
            except Exception as e:
                logger.warning(f"verify_math_answer failed: {e}, defaulting reward=0.0")
                reward = 0.0
            samples.append(RolloutSample(
                prompt=prompt,
                response=resp,
                reward=reward,
                layer=layer,
            ))
        return samples

    def _tokenize_samples(
        self,
        samples: List[RolloutSample],
    ) -> List[RolloutSample]:
        """
        Tokenize và gán token_ids cho mỗi sample.
        Gọi sau khi training engine đã được init (có tokenizer).
        """
        tokenizer = self.training_engine.tokenizer
        for sample in samples:
            # Format: prompt + response
            full_text = (
                f"<|im_start|>user\n{sample.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{sample.response}<|im_end|>"
            )
            encoded = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
            )
            sample.token_ids = encoded["input_ids"].squeeze(0)         # (L,)
            sample.attention_mask = encoded["attention_mask"].squeeze(0)  # (L,)
        return samples

    def _run_layer1(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> List[List[RolloutSample]]:
        """
        Giai đoạn 1: Sinh mẫu Layer 1 và đánh giá.

        Returns:
            List[List[RolloutSample]] — nhóm theo prompt (len=B, mỗi nhóm len=N).
        """
        logger.info(f"=== LAYER 1: Generating {self.n_rollouts} rollouts per prompt ===")

        # Sinh mẫu bằng vLLM
        all_responses = self.vllm_engine.generate(
            prompts=prompts,
            n=self.n_rollouts,
        )  # List[List[str]], len=B, inner len=N

        # Tạo RolloutSample và đánh giá
        batch_samples = []
        for prompt, responses, gt in zip(prompts, all_responses, ground_truths):
            samples = self._create_rollout_samples(prompt, responses, gt, layer=1)
            batch_samples.append(samples)

        return batch_samples

    def _run_layer2(
        self,
        prompt: str,
        layer1_samples: List[RolloutSample],
        ground_truth: str,
    ) -> List[RolloutSample]:
        """
        Giai đoạn 2: Self-Reflection & Augmented Sampling.

        Mỗi mẫu Layer 1 được sinh M nhánh suy luận mới tại Layer 2.

        Returns:
            List[RolloutSample] — tập mẫu mở rộng (N*M mẫu).
        """
        logger.info(f"=== LAYER 2: Self-reflection ({self.m_reflections} branches/sample) ===")

        # Xây dựng reflection prompts: mỗi mẫu Layer 1 trở thành input cho Layer 2
        reflection_prompts = []
        for sample in layer1_samples:
            instruction = random.choice(SELF_REFLECTION_INSTRUCTIONS)
            reflection_prompt = (
                f"Original problem: {sample.prompt}\n\n"
                f"Previous solution:\n{sample.response}\n\n"
                f"{instruction}"
            )
            reflection_prompts.append(reflection_prompt)

        # Sinh M nhánh per reflection prompt
        all_responses = self.vllm_engine.generate(
            prompts=reflection_prompts,
            n=self.m_reflections,
            system_prompt=SELF_REFLECTION_SYSTEM_PROMPT,
        )  # List[List[str]], len=N, inner len=M

        # Tạo expanded sample pool
        expanded_samples = []
        for parent_idx, (responses, parent) in enumerate(
            zip(all_responses, layer1_samples)
        ):
            for resp in responses:
                reward = verify_math_answer(resp, ground_truth)
                expanded_samples.append(RolloutSample(
                    prompt=prompt,
                    response=resp,
                    reward=reward,
                    layer=2,
                    parent_index=parent_idx,
                ))

        logger.info(f"Layer 2 produced {len(expanded_samples)} samples "
                    f"(N={len(layer1_samples)} x M={self.m_reflections})")

        return expanded_samples

    def train_on_prompt(
        self,
        prompt: str,
        ground_truth: str,
        layer1_samples: List[RolloutSample],
        layer2_samples: Optional[List[RolloutSample]] = None,
    ) -> Dict[str, float]:
        """
        Xử lý Dynamic Routing và huấn luyện cho một prompt.
        """
        # ---- Tính Correct Ratio Layer 1 ----
        if not layer1_samples:
            logger.warning("Layer 1 returned 0 samples, skipping this prompt.")
            self.global_step += 1
            return {"total_loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
                    "entropy_scale": 1.0, "kl_penalty_mean": 0.0,
                    "correct_ratio": 0.0, "mean_advantage": 0.0,
                    "routing": "SKIP", "step": self.global_step - 1}

        correct_pool_l1, error_pool_l1, correct_ratio = evaluate_and_split_pools(
            layer1_samples
        )

        logger.info(f"Layer 1 correct ratio: {correct_ratio:.2%} "
                    f"({len(correct_pool_l1)} correct / {len(error_pool_l1)} errors)")

        # ---- Conditioning Gate ----
        is_extreme = (correct_ratio == 0.0) or (correct_ratio == 1.0)

        if is_extreme:
            # ========== NHÁNH 1: NGRPO (batch cực đoan) ==========
            logger.info("→ Routing to NGRPO (extreme batch, skipping Layer 2)")

            rng = np.random.RandomState(self.global_step)
            indices = rng.choice(len(layer1_samples), size=self.batch_size_k, replace=True)
            ngrpo_samples = [layer1_samples[i] for i in indices]

            ngrpo_samples = self._tokenize_samples(ngrpo_samples)

            batch_tensor = collate_batch_to_tensors(
                batch_result=BatchResult(
                    correct_samples=ngrpo_samples if correct_ratio == 1.0 else [],
                    error_samples=ngrpo_samples if correct_ratio == 0.0 else [],
                    cluster_distribution=torch.ones(self.n_clusters) / self.n_clusters,
                    top_cluster_id=0,
                    correct_ratio=correct_ratio,
                ),
                max_seq_len=self.max_seq_len,
                pad_token_id=self.training_engine.tokenizer.pad_token_id,
            )

        else:
            # ========== NHÁNH 2: Full Pipeline (Layer 1 → Layer 2) ==========
            logger.info("→ Routing to Layer 2 (mixed batch)")

            if layer2_samples is None or len(layer2_samples) == 0:
                logger.warning("layer2_samples is empty/None for mixed prompt, falling back to L1 NGRPO")
                rng = np.random.RandomState(self.global_step)
                indices = rng.choice(len(layer1_samples), size=self.batch_size_k, replace=True)
                ngrpo_samples = [layer1_samples[i] for i in indices]
                ngrpo_samples = self._tokenize_samples(ngrpo_samples)
                batch_tensor = collate_batch_to_tensors(
                    batch_result=BatchResult(
                        correct_samples=ngrpo_samples,
                        error_samples=[],
                        cluster_distribution=torch.ones(self.n_clusters) / self.n_clusters,
                        top_cluster_id=0,
                        correct_ratio=correct_ratio,
                    ),
                    max_seq_len=self.max_seq_len,
                    pad_token_id=self.training_engine.tokenizer.pad_token_id,
                )
                is_extreme = True
            else:
                # ---- Evaluate & Split ----
                correct_pool, error_pool, l2_correct_ratio = evaluate_and_split_pools(
                    layer2_samples
                )
                logger.info(f"Layer 2 correct ratio: {l2_correct_ratio:.2%}")

                # Fallback: nếu sau Layer 2 toàn bộ đúng hoặc toàn bộ sai
                # → chuyển sang NGRPO-style (dùng toàn bộ pool có sẵn)
                l2_is_extreme = (l2_correct_ratio == 0.0) or (l2_correct_ratio == 1.0)

                if l2_is_extreme:
                    logger.info(f"→ Layer 2 extreme ({l2_correct_ratio:.0%}), fallback to NGRPO")
                    available_pool = correct_pool if correct_pool else error_pool
                    rng = np.random.RandomState(self.global_step)
                    indices = rng.choice(len(available_pool), size=self.batch_size_k, replace=True)
                    ngrpo_samples = [available_pool[i] for i in indices]
                    ngrpo_samples = self._tokenize_samples(ngrpo_samples)

                    batch_tensor = collate_batch_to_tensors(
                        batch_result=BatchResult(
                            correct_samples=ngrpo_samples if l2_correct_ratio == 1.0 else [],
                            error_samples=ngrpo_samples if l2_correct_ratio == 0.0 else [],
                            cluster_distribution=torch.ones(self.n_clusters) / self.n_clusters,
                            top_cluster_id=0,
                            correct_ratio=l2_correct_ratio,
                        ),
                        max_seq_len=self.max_seq_len,
                        pad_token_id=self.training_engine.tokenizer.pad_token_id,
                    )
                    # Override routing label
                    is_extreme = True
                else:
                    # ---- Semantic Error Profiling ----
                    labels, embeddings, cluster_dist, top_cluster_id = \
                        self.error_profiler.cluster_errors(error_pool)

                    logger.info(f"Error clusters: {self.n_clusters}, "
                                f"top cluster ID: {top_cluster_id}, "
                                f"distribution: {cluster_dist.tolist()}")

                    # ---- Static Balanced Batch ----
                    batch_result = build_static_balanced_batch(
                        correct_pool=correct_pool,
                        error_pool=error_pool,
                        cluster_labels=labels,
                        top_cluster_id=top_cluster_id,
                        cluster_distribution=cluster_dist,
                        K=self.batch_size_k,
                        seed=self.global_step,
                    )

                    # Tokenize
                    all_batch_samples = batch_result.correct_samples + batch_result.error_samples
                    all_batch_samples = self._tokenize_samples(all_batch_samples)
                    batch_result.correct_samples = all_batch_samples[:self.batch_size_k // 2]
                    batch_result.error_samples = all_batch_samples[self.batch_size_k // 2:]

                    batch_tensor = collate_batch_to_tensors(
                        batch_result=batch_result,
                        max_seq_len=self.max_seq_len,
                        pad_token_id=self.training_engine.tokenizer.pad_token_id,
                    )

        # ---- Training Step ----
        metrics = self.training_engine.training_step(
            batch=batch_tensor,
            kl_coeff=self.kl_coeff,
            clip_eps=self.clip_eps,
            virtual_reward=self.virtual_reward,
            entropy_scale_min=self.entropy_scale_min,
        )

        metrics["routing"] = "NGRPO" if is_extreme else "Layer2"
        metrics["step"] = self.global_step
        self.global_step += 1
        self.metrics_log.append(metrics)

        logger.info(f"Step {metrics['step']} | Route: {metrics['routing']} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"Scale: {metrics['entropy_scale']:.4f}")

        return metrics

    def train(
        self,
        dataset: List[Dict[str, str]],
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        resume: bool = False,
    ):
        """
        Vòng lặp huấn luyện chính với hỗ trợ resume từ checkpoint.

        Args:
            dataset: List[Dict] — mỗi dict chứa {"prompt": str, "answer": str}.
            num_epochs: int — số epoch.
            resume: bool — nếu True, tự động tìm và load checkpoint gần nhất.
        """
        logger.info("=" * 60)
        logger.info("A-MSB-GRPO Training Pipeline Starting")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"N (Layer 1 rollouts): {self.n_rollouts}")
        logger.info(f"M (Layer 2 reflections): {self.m_reflections}")
        logger.info(f"K (static batch size): {self.batch_size_k}")
        logger.info(f"Resume: {resume}")
        logger.info("=" * 60)

        # ---- Init Training Engine ----
        self.training_engine.init_model()
        if self.training_engine.deepspeed_config is not None:
            self.training_engine.init_deepspeed()
        else:
            self.training_engine.init_standalone()

        # ---- Resume from checkpoint if requested ----
        if resume:
            ckpt_dir = self._find_latest_checkpoint()
            if ckpt_dir:
                self._load_checkpoint(ckpt_dir)
            else:
                logger.info("No checkpoint found, starting from scratch.")

        total_steps = num_epochs * len(dataset)
        train_start_time = time.time()

        # ---- Epoch Progress Bar ----
        epoch_pbar = tqdm(
            range(num_epochs),
            desc="🔄 Epochs",
            unit="epoch",
            position=0,
            leave=True,
            colour="blue",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        for epoch in epoch_pbar:
            # Skip đã hoàn thành epochs khi resume
            if epoch < self._resume_epoch:
                logger.info(f"Skipping epoch {epoch+1} (already completed)")
                continue

            epoch_pbar.set_description(f"🔄 Epoch {epoch+1}/{num_epochs}")
            logger.info(f"\n{'='*40} EPOCH {epoch+1}/{num_epochs} {'='*40}")

            epoch_losses = []
            epoch_start_time = time.time()

            # Xác định vị trí bắt đầu trong epoch này
            start_prompt_idx = 0
            if epoch == self._resume_epoch and self._resume_prompt_idx > 0:
                start_prompt_idx = self._resume_prompt_idx
                logger.info(f"Resuming epoch {epoch+1} from prompt index {start_prompt_idx}")

            # ---- Prompt Progress Bar ----
            prompt_pbar = tqdm(
                total=len(dataset),
                initial=start_prompt_idx,
                desc="Training",
                unit="prompt",
                position=1,
                leave=False,
                colour="green",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

            prompt_batch_size = 16

            for batch_start in range(start_prompt_idx, len(dataset), prompt_batch_size):
                batch_end = min(batch_start + prompt_batch_size, len(dataset))
                batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

                prompts = [s["prompt"] for s in batch_samples]
                ground_truths = [s["answer"] for s in batch_samples]

                # ---- Flush VRAM trước khi vLLM khởi tạo ----
                flush_vram()

                # ---- Layer 1: Generate cho toàn bộ batch ----
                layer1_groups = self._run_layer1(
                    prompts=prompts,
                    ground_truths=ground_truths,
                )

                # Xác định routing và chạy Layer 2 nếu cần cho từng prompt
                layer2_groups_dict = {}
                is_extreme_list = []

                for b_idx in range(len(prompts)):
                    try:
                        l1_samples = layer1_groups[b_idx]
                        _, _, correct_ratio = evaluate_and_split_pools(l1_samples)
                        is_extreme = (correct_ratio == 0.0) or (correct_ratio == 1.0)
                        is_extreme_list.append(is_extreme)

                        if not is_extreme:
                            # Gọi Layer 2 khi vLLM engine vẫn còn sống
                            l2_samples = self._run_layer2(prompts[b_idx], l1_samples, ground_truths[b_idx])
                            layer2_groups_dict[b_idx] = l2_samples
                    except Exception as e:
                        logger.warning(f"Layer 2 failed for prompt {batch_start + b_idx}: {e}, will fallback")
                        is_extreme_list.append(True)  # fallback đến NGRPO

                # ---- Giải phóng vLLM trước khi train ----
                self.vllm_engine.destroy()
                flush_vram()

                # ---- Train bằng PyTorch cho từng prompt ----
                for b_idx in range(len(prompts)):
                    try:
                        metrics = self.train_on_prompt(
                            prompt=prompts[b_idx],
                            ground_truth=ground_truths[b_idx],
                            layer1_samples=layer1_groups[b_idx],
                            layer2_samples=layer2_groups_dict.get(b_idx, None)
                        )
                    except Exception as e:
                        logger.error(f"Training failed for prompt {batch_start + b_idx}: {e}")
                        logger.error("Skipping this prompt and continuing...")
                        self.global_step += 1
                        metrics = {"total_loss": 0.0, "routing": "ERROR",
                                   "entropy_scale": 0, "step": self.global_step - 1}

                    # ---- Cập nhật Progress Bar ----
                    epoch_losses.append(metrics["total_loss"])
                    avg_loss = sum(epoch_losses) / len(epoch_losses)

                    prompt_pbar.update(1)

                    prompt_pbar.set_postfix({
                        "loss": f"{metrics['total_loss']:.4f}",
                        "avg": f"{avg_loss:.4f}",
                        "route": metrics.get('routing', 'N/A'),
                    })

                    # Periodic checkpoint (mỗi 100 steps)
                    if (self.global_step % 100 == 0) and (self.global_step > 0):
                        self._save_checkpoint(epoch, self.global_step, prompt_idx=batch_end)

                # Checkpoint sau mỗi prompt-batch (để resume chính xác)
                self._save_checkpoint(epoch, self.global_step, prompt_idx=batch_end)

            # Reset resume position cho epoch tiếp theo
            self._resume_prompt_idx = 0

            prompt_pbar.close()

            # ---- Epoch Summary ----
            epoch_time = time.time() - epoch_start_time
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            epoch_pbar.set_postfix({
                "avg_loss": f"{epoch_avg_loss:.4f}",
                "time": f"{epoch_time/60:.1f}min",
            }, refresh=True)

            logger.info(f"Epoch {epoch+1} complete | Avg Loss: {epoch_avg_loss:.4f} | "
                        f"Time: {epoch_time/60:.1f} min")

            # End epoch checkpoint
            self._save_checkpoint(epoch, self.global_step, prompt_idx=len(dataset))

        epoch_pbar.close()

        total_time = time.time() - train_start_time
        logger.info("\n" + "=" * 60)
        logger.info("✅ Training Complete!")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        self._save_metrics()

    def _save_checkpoint(self, epoch: int, step: int, prompt_idx: int = 0):
        """
        Lưu checkpoint đầy đủ trạng thái training để có thể resume.
        Bao gồm: LoRA weights, optimizer state, epoch, step, vị trí dataset.
        """
        ckpt_dir = os.path.join(self.output_dir, f"epoch{epoch}_step{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save LoRA adapter weights
        if self.training_engine.model is not None:
            self.training_engine.model.save_pretrained(ckpt_dir)

        # 2. Save optimizer state
        if self.training_engine.optimizer is not None:
            torch.save(
                self.training_engine.optimizer.state_dict(),
                os.path.join(ckpt_dir, "optimizer.pt")
            )

        # 3. Save training state metadata
        state = {
            "epoch": epoch,
            "global_step": step,
            "prompt_idx": prompt_idx,
            "metrics_log": self.metrics_log,
        }
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        # 4. Symbolic link to latest checkpoint
        latest_link = os.path.join(self.output_dir, "latest")
        if os.path.islink(latest_link):
            os.remove(latest_link)
        elif os.path.exists(latest_link):
            import shutil
            shutil.rmtree(latest_link, ignore_errors=True)
        os.symlink(os.path.abspath(ckpt_dir), latest_link)

        logger.info(f"Checkpoint saved: {ckpt_dir} (epoch={epoch}, step={step}, prompt_idx={prompt_idx})")

        # 5. Clean up old checkpoints (Keep only newest 3)
        import shutil
        checkpoint_dirs = [
            os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir)
            if d.startswith("epoch") and os.path.isdir(os.path.join(self.output_dir, d))
        ]
        # Sort by creation/modification time
        checkpoint_dirs.sort(key=os.path.getmtime)
        
        max_checkpoints = 3
        if len(checkpoint_dirs) > max_checkpoints:
            for old_ckpt in checkpoint_dirs[:-max_checkpoints]:
                logger.info(f"Removing old checkpoint: {old_ckpt} to save disk space")
                shutil.rmtree(old_ckpt, ignore_errors=True)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Tìm checkpoint mới nhất từ thư mục output."""
        latest_link = os.path.join(self.output_dir, "latest")
        if os.path.exists(latest_link):
            resolved = os.path.realpath(latest_link)
            state_file = os.path.join(resolved, "training_state.json")
            if os.path.exists(state_file):
                return resolved
        return None

    def _load_checkpoint(self, ckpt_dir: str):
        """
        Load checkpoint và khôi phục trạng thái training.
        Gọi sau khi training_engine.đã init model và optimizer.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 RESUMING from checkpoint: {ckpt_dir}")
        logger.info(f"{'='*60}")

        # 1. Load LoRA adapter weights
        from peft import PeftModel
        adapter_config = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.exists(adapter_config):
            self.training_engine.model.load_adapter(ckpt_dir, adapter_name="default")
            logger.info(f"  ✔ LoRA weights loaded")

        # 2. Load optimizer state
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path) and self.training_engine.optimizer is not None:
            self.training_engine.optimizer.load_state_dict(
                torch.load(opt_path, map_location="cuda")
            )
            logger.info(f"  ✔ Optimizer state loaded")

        # 3. Load training state metadata
        state_path = os.path.join(ckpt_dir, "training_state.json")
        with open(state_path, "r") as f:
            state = json.load(f)

        self.global_step = state["global_step"]
        self._resume_epoch = state["epoch"]
        self._resume_prompt_idx = state["prompt_idx"]
        self.metrics_log = state.get("metrics_log", [])

        logger.info(f"  ✔ Resuming from epoch={self._resume_epoch}, "
                    f"step={self.global_step}, prompt_idx={self._resume_prompt_idx}")
        logger.info(f"{'='*60}\n")

    def _save_metrics(self):
        """Lưu toàn bộ metrics log ra JSON."""
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")
