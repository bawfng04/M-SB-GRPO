"""
A-MSB-GRPO Training Entry Point
=================================

Script khởi chạy pipeline huấn luyện A-MSB-GRPO cho Qwen-2.5-7B-Instruct.

Hỗ trợ 2 nguồn dữ liệu:
  1. Parquet files từ prepare_requested_datasets.py (ưu tiên)
  2. JSONL files (fallback)

Cách chạy:
    # Với Parquet datasets (từ prepare_requested_datasets.py)
    deepspeed run_train.py \\
        --deepspeed configs/ds_config_zero3.json \\
        --train-parquet ../../data/datasets/requested/train/combined_train.parquet \\
        --test-parquet ../../data/datasets/requested/test/combined_test.parquet

    # Với JSONL (legacy)
    deepspeed run_train.py --deepspeed configs/ds_config_zero3.json --data-path ./data/math_train.jsonl

    # Debug (không DeepSpeed)
    python3 run_train.py --no-deepspeed --max-samples 5
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Optional

from src.train import AMSBGRPOTrainer

logger = logging.getLogger(__name__)


# ===========================================================================
# Dataset Loading — hỗ trợ Parquet + JSONL
# ===========================================================================

def load_parquet_dataset(parquet_path: str, max_samples: int = -1) -> List[Dict[str, str]]:
    """
    Load dataset từ Parquet file (output của prepare_requested_datasets.py).

    Parquet schema: problem, solution, answer_extracted, source_dataset, source_split, metadata

    Args:
        parquet_path: str — đường dẫn tới file .parquet.
        max_samples: int — giới hạn mẫu (-1 = toàn bộ).

    Returns:
        List[Dict[str, str]] — list {"prompt": ..., "answer": ...}.
    """
    import pyarrow.parquet as pq

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    logger.info(f"Loaded parquet: {parquet_path} ({len(df)} rows)")
    logger.info(f"Columns: {list(df.columns)}")

    dataset = []
    for _, row in df.iterrows():
        # Map từ parquet schema sang pipeline schema
        prompt = str(row.get("problem", ""))
        # Ưu tiên answer_extracted (đã trích xuất giá trị cuối),
        # fallback sang solution (lời giải đầy đủ)
        answer = str(row.get("answer_extracted", row.get("solution", "")))

        if prompt and answer:
            dataset.append({
                "prompt": prompt,
                "answer": answer,
                # Giữ metadata cho logging
                "source_dataset": str(row.get("source_dataset", "")),
                "solution_full": str(row.get("solution", "")),
            })

    if max_samples > 0:
        dataset = dataset[:max_samples]

    logger.info(f"Prepared {len(dataset)} samples from parquet")
    return dataset


def load_jsonl_dataset(data_path: str, max_samples: int = -1) -> List[Dict[str, str]]:
    """
    Load dataset từ JSONL file (legacy format).

    JSONL schema: {"problem": "...", "solution": "..."} hoặc {"prompt": "...", "answer": "..."}
    """
    dataset = []

    if not os.path.exists(data_path):
        logger.warning(f"Dataset file not found: {data_path}")
        logger.info("Generating dummy dataset for testing...")
        dataset = [
            {"prompt": "Solve: What is the value of $2^{10}$?", "answer": "\\boxed{1024}"},
            {"prompt": "Find $\\frac{1}{2} + \\frac{1}{3}$.", "answer": "\\boxed{\\frac{5}{6}}"},
            {"prompt": "If $f(x) = x^2 + 3x + 2$, what is $f(1)$?", "answer": "\\boxed{6}"},
        ]
        return dataset

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dataset.append({
                "prompt": item.get("problem", item.get("prompt", "")),
                "answer": item.get("solution", item.get("answer", "")),
            })

    if max_samples > 0:
        dataset = dataset[:max_samples]

    logger.info(f"Loaded {len(dataset)} samples from {data_path}")
    return dataset


def load_dataset(args) -> tuple:
    """
    Load train và test dataset dựa trên args.

    Returns:
        (train_dataset, test_dataset) — test_dataset có thể là None nếu không chỉ định.
    """
    train_dataset = None
    test_dataset = None

    # ---- Train dataset ----
    if args.train_parquet:
        train_dataset = load_parquet_dataset(args.train_parquet, args.max_samples)
    elif args.data_path:
        train_dataset = load_jsonl_dataset(args.data_path, args.max_samples)
    else:
        logger.warning("No train data specified! Using dummy dataset.")
        train_dataset = load_jsonl_dataset("__nonexistent__", args.max_samples)

    # ---- Test dataset ----
    if args.test_parquet:
        test_dataset = load_parquet_dataset(args.test_parquet, args.max_test_samples)
    elif args.test_path:
        test_dataset = load_jsonl_dataset(args.test_path, args.max_test_samples)

    return train_dataset, test_dataset


# ===========================================================================
# CLI Arguments
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="A-MSB-GRPO Training Pipeline for Qwen-2.5-7B-Instruct"
    )

    # === Data Sources ===
    data_group = parser.add_argument_group("Data Sources")
    # Parquet (từ prepare_requested_datasets.py — ưu tiên)
    data_group.add_argument("--train-parquet", type=str, default=None,
                            help="Path to train parquet file (from prepare_requested_datasets.py).")
    data_group.add_argument("--test-parquet", type=str, default=None,
                            help="Path to test parquet file (from prepare_requested_datasets.py).")
    # JSONL (legacy fallback)
    data_group.add_argument("--data-path", type=str, default=None,
                            help="Path to JSONL train file (legacy).")
    data_group.add_argument("--test-path", type=str, default=None,
                            help="Path to JSONL test file (legacy).")
    data_group.add_argument("--max-samples", type=int, default=-1,
                            help="Max number of training samples (-1 = all).")
    data_group.add_argument("--max-test-samples", type=int, default=-1,
                            help="Max number of test samples (-1 = all).")

    # === Model ===
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                             help="HuggingFace model name or local path.")
    model_group.add_argument("--use-qlora", action="store_true", default=True,
                             help="Use QLoRA (4-bit quantization).")
    model_group.add_argument("--no-qlora", action="store_false", dest="use_qlora",
                             help="Disable QLoRA, use full-precision LoRA.")
    model_group.add_argument("--lora-r", type=int, default=16)
    model_group.add_argument("--lora-alpha", type=int, default=32)

    # === A-MSB-GRPO Hyperparameters ===
    algo_group = parser.add_argument_group("A-MSB-GRPO Hyperparameters")
    algo_group.add_argument("--n-rollouts", type=int, default=8,
                            help="N: number of Layer 1 rollouts per prompt.")
    algo_group.add_argument("--m-reflections", type=int, default=4,
                            help="M: number of Layer 2 reflection branches per rollout.")
    algo_group.add_argument("--batch-size-k", type=int, default=16,
                            help="K: static balanced batch size.")
    algo_group.add_argument("--n-clusters", type=int, default=4,
                            help="Number of K-Means clusters for error profiling.")

    # === Loss Function ===
    loss_group = parser.add_argument_group("Loss Function")
    loss_group.add_argument("--kl-coeff", type=float, default=0.01,
                            help="KL divergence penalty coefficient (β).")
    loss_group.add_argument("--clip-eps", type=float, default=0.2,
                            help="Clipping epsilon for policy ratio.")
    loss_group.add_argument("--virtual-reward", type=float, default=-1.0,
                            help="Virtual reward value for NGRPO.")
    loss_group.add_argument("--entropy-scale-min", type=float, default=0.01,
                            help="Minimum entropy scale floor.")

    # === Training ===
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num-epochs", type=int, default=3)
    train_group.add_argument("--learning-rate", type=float, default=1e-5)
    train_group.add_argument("--resume", action="store_true", default=False,
                             help="Resume training from latest checkpoint.")
    train_group.add_argument("--max-new-tokens", type=int, default=2048,
                             help="Max tokens for vLLM generation.")
    train_group.add_argument("--max-seq-len", type=int, default=4096,
                             help="Max sequence length for tokenization.")

    # === vLLM ===
    vllm_group = parser.add_argument_group("vLLM Engine")
    vllm_group.add_argument("--vllm-gpu-util", type=float, default=0.45,
                            help="GPU memory utilization for vLLM (0.0-1.0).")

    # === DeepSpeed ===
    ds_group = parser.add_argument_group("DeepSpeed")
    ds_group.add_argument("--deepspeed", type=str, default=None,
                          help="Path to DeepSpeed config JSON.")
    ds_group.add_argument("--no-deepspeed", action="store_true",
                          help="Disable DeepSpeed (for debugging).")
    ds_group.add_argument("--local_rank", type=int, default=-1,
                          help="Local rank for distributed training (set by DeepSpeed).")

    # === Output ===
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default="./checkpoints",
                           help="Directory for checkpoints and metrics.")
    out_group.add_argument("--log-file", type=str, default=None,
                           help="Path to log file. Logs go to file only (not terminal).")

    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # ---- Setup logging: file only (stderr clean for tqdm) ----
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    if args.log_file:
        import os
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        # Root logger → file only
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # Clear console handlers
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        # Suppress noisy third-party loggers
        for noisy in ["vllm", "transformers", "torch", "accelerate", "peft",
                      "sentence_transformers", "urllib3", "filelock", "huggingface_hub"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    else:
        # Fallback: log to stdout (for debugging)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
        )

    logger.info("=" * 60)
    logger.info("A-MSB-GRPO Training Pipeline")
    logger.info("=" * 60)

    # ---- Load DeepSpeed config ----
    ds_config = None
    if args.deepspeed and not args.no_deepspeed:
        with open(args.deepspeed, "r") as f:
            ds_config = json.load(f)
        logger.info(f"DeepSpeed config loaded from: {args.deepspeed}")

    # ---- Load Datasets ----
    train_dataset, test_dataset = load_dataset(args)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    if test_dataset:
        logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Log dataset source breakdown
    sources = {}
    for item in train_dataset:
        src = item.get("source_dataset", "unknown")
        sources[src] = sources.get(src, 0) + 1
    if sources:
        logger.info(f"Train dataset breakdown: {sources}")

    # ---- Initialize Trainer ----
    trainer = AMSBGRPOTrainer(
        model_name=args.model_name,
        n_rollouts=args.n_rollouts,
        m_reflections=args.m_reflections,
        batch_size_k=args.batch_size_k,
        n_clusters=args.n_clusters,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        kl_coeff=args.kl_coeff,
        clip_eps=args.clip_eps,
        virtual_reward=args.virtual_reward,
        entropy_scale_min=args.entropy_scale_min,
        vllm_gpu_util=args.vllm_gpu_util,
        use_qlora=args.use_qlora,
        deepspeed_config=ds_config,
        output_dir=args.output_dir,
    )

    # ---- Train ----
    trainer.train(dataset=train_dataset, num_epochs=args.num_epochs, resume=args.resume)


if __name__ == "__main__":
    main()
