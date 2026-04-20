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
from typing import Dict, List, Optional, Any

import torch
import numpy as np

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
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        gpu_memory_utilization: float = 0.45,
        tensor_parallel_size: int = 1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
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
            max_model_len=DEFAULT_MAX_SEQ_LEN,
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
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.deepspeed_config = deepspeed_config
        self.use_qlora = use_qlora

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
                    "device": "none",
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

        Args:
            input_ids:      (K, L) — token IDs.
            attention_mask: (K, L) — attention mask.
            use_ref: bool — nếu True, disable LoRA adapter để tính ref logprobs.

        Returns:
            logprobs: (K, L) — per-token log probabilities.

        Ghi chú Shape:
            logits (from model) → (K, L, V)
            logprobs            → (K, L)
        """
        if use_ref:
            # Disable LoRA adapter → reference model output
            self.model.disable_adapter_layers()

        with torch.no_grad() if use_ref else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (K, L, V)

        logprobs = compute_per_token_logprobs(logits, input_ids)  # (K, L)

        if use_ref:
            # Re-enable LoRA adapter
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
        Một bước huấn luyện hoàn chỉnh: forward → loss → backward.

        Args:
            batch: Dict — output từ collate_batch_to_tensors().
            kl_coeff, clip_eps, virtual_reward, entropy_scale_min: hyperparams.

        Returns:
            Dict[str, float] — metrics cho logging.

        Luồng dữ liệu:
            1. Forward pass (current policy) → current_logprobs (K, L)
            2. Forward pass (ref policy, LoRA off) → ref_logprobs (K, L)
            3. Tính A-MSB-GRPO Loss
            4. Backward pass
        """
        input_ids = batch["input_ids"]            # (K, L)
        attention_mask = batch["attention_mask"]   # (K, L)
        old_logprobs = batch["old_logprobs"]       # (K, L)
        rewards = batch["rewards"]                 # (K,)
        correct_ratio = batch["correct_ratio"]     # ()
        cluster_dist = batch["cluster_dist"]       # (C,)

        # ---- Bước 1: Forward pass (current policy → trainable) ----
        current_logprobs = self.compute_logprobs_for_batch(
            input_ids, attention_mask, use_ref=False
        )  # (K, L)

        # ---- Bước 2: Forward pass (reference policy → frozen / LoRA off) ----
        ref_logprobs = self.compute_logprobs_for_batch(
            input_ids, attention_mask, use_ref=True
        )  # (K, L)

        # ---- Bước 3: Tính Loss ----
        loss, info = amsb_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards,
            attention_mask=attention_mask,
            correct_ratio=correct_ratio,
            cluster_distribution=cluster_dist,
            virtual_reward=virtual_reward,
            kl_coeff=kl_coeff,
            clip_eps=clip_eps,
            entropy_scale_min=entropy_scale_min,
        )

        # ---- Bước 4: Backward pass ----
        if self._ds_engine is not None:
            self._ds_engine.backward(loss)
            self._ds_engine.step()
        else:
            loss.backward()
            if self.optimizer:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # ---- Metrics ----
        metrics = {
            "total_loss": info["total_loss"].item(),
            "policy_loss": info["policy_loss"].item(),
            "kl_loss": info["kl_loss"].item(),
            "entropy_scale": info["entropy_scale"].item(),
            "kl_penalty_mean": info["kl_penalty_mean"].item(),
            "correct_ratio": info["correct_ratio"].item(),
            "mean_advantage": info["advantages"].mean().item(),
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

    def _create_rollout_samples(
        self,
        prompt: str,
        responses: List[str],
        ground_truth: str,
        layer: int = 1,
    ) -> List[RolloutSample]:
        """Tạo RolloutSample từ output vLLM."""
        samples = []
        for resp in responses:
            reward = verify_math_answer(resp, ground_truth)
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
            reflection_prompt = (
                f"Bài toán gốc: {sample.prompt}\n\n"
                f"Lời giải trước đó:\n{sample.response}\n\n"
                f"Hãy kiểm tra lại lời giải trên và đưa ra đáp án cuối cùng."
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
    ) -> Dict[str, float]:
        """
        Xử lý Dynamic Routing và huấn luyện cho một prompt.

        Luồng xử lý:
          1. Tính correct_ratio từ Layer 1.
          2. Conditioning Gate:
             - 0% hoặc 100% → NGRPO trực tiếp (batch tĩnh K, cùng một label).
             - Mixed → Layer 2 → Semantic Profiling → Balanced Batch.
          3. Training step.

        Returns:
            Dict[str, float] — training metrics.
        """
        # ---- Tính Correct Ratio Layer 1 ----
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

            # Dùng trực tiếp mẫu Layer 1 làm training batch
            # Sampling with replacement lên đúng K mẫu
            rng = np.random.RandomState(self.global_step)
            indices = rng.choice(len(layer1_samples), size=self.batch_size_k, replace=True)
            ngrpo_samples = [layer1_samples[i] for i in indices]

            # Tokenize
            ngrpo_samples = self._tokenize_samples(ngrpo_samples)

            # Cần tạo BatchResult với cluster_dist = None (sẽ scale = 1.0)
            rewards_tensor = torch.tensor([s.reward for s in ngrpo_samples])
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

            # Ghi đè cluster_dist = uniform (Scale = 1.0 cho NGRPO)
            # correct_ratio = 0.0 hoặc 1.0 → gate tự động chọn NGRPO advantage

        else:
            # ========== NHÁNH 2: Full Pipeline (Layer 1 → Layer 2) ==========
            logger.info("→ Routing to Layer 2 (mixed batch)")

            # ---- Layer 2: Self-Reflection ----
            layer2_samples = self._run_layer2(prompt, layer1_samples, ground_truth)

            # ---- Evaluate & Split ----
            correct_pool, error_pool, l2_correct_ratio = evaluate_and_split_pools(
                layer2_samples
            )
            logger.info(f"Layer 2 correct ratio: {l2_correct_ratio:.2%}")

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

        # ---- Giải phóng vLLM trước khi train ----
        self.vllm_engine.destroy()

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
    ):
        """
        Vòng lặp huấn luyện chính.

        Args:
            dataset: List[Dict] — mỗi dict chứa {"prompt": str, "answer": str}.
            num_epochs: int — số epoch.
        """
        logger.info("=" * 60)
        logger.info("A-MSB-GRPO Training Pipeline Starting")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"N (Layer 1 rollouts): {self.n_rollouts}")
        logger.info(f"M (Layer 2 reflections): {self.m_reflections}")
        logger.info(f"K (static batch size): {self.batch_size_k}")
        logger.info("=" * 60)

        # ---- Init Training Engine ----
        self.training_engine.init_model()
        self.training_engine.init_deepspeed()

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*40} EPOCH {epoch+1}/{num_epochs} {'='*40}")

            for idx, sample in enumerate(dataset):
                prompt = sample["prompt"]
                ground_truth = sample["answer"]

                logger.info(f"\n--- Prompt {idx+1}/{len(dataset)} ---")

                # ---- Layer 1: Generate ----
                layer1_groups = self._run_layer1(
                    prompts=[prompt],
                    ground_truths=[ground_truth],
                )
                layer1_samples = layer1_groups[0]  # chỉ 1 prompt tại thời điểm

                # ---- Dynamic Routing & Train ----
                metrics = self.train_on_prompt(prompt, ground_truth, layer1_samples)

                # Periodic checkpoint
                if (self.global_step % 100 == 0) and (self.global_step > 0):
                    self._save_checkpoint(epoch, self.global_step)

            # End epoch checkpoint
            self._save_checkpoint(epoch, self.global_step)

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total steps: {self.global_step}")
        self._save_metrics()

    def _save_checkpoint(self, epoch: int, step: int):
        """Lưu checkpoint (LoRA adapter weights)."""
        ckpt_dir = os.path.join(self.output_dir, f"epoch{epoch}_step{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.training_engine.model is not None:
            self.training_engine.model.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _save_metrics(self):
        """Lưu toàn bộ metrics log ra JSON."""
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")
