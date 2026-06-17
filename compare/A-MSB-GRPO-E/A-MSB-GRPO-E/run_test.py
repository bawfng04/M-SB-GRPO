#!/usr/bin/env python3
"""
A-MSB-GRPO Test Inference & Evaluation Generation Pipeline
==========================================================
Sinh các mẫu test sử dụng vLLM có tích hợp checkpoint LoRA mới nhất,
sau đó định dạng output thành JSONL cho test.sh đánh giá.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark outputs via vLLM")
    parser.add_argument("--test-parquet", type=str, required=True, help="Test dataset path")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-dir", type=str, required=True, help="Path to latest LoRA checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save JSONL output")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to generate per prompt (pass@K)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def extract_boxed(text: str) -> str:
    """Trích xuất nội dung trong \\boxed{...} hoặc <answer>...</answer>"""
    import re
    # Thử lấy từ XML tag trước
    xml_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.DOTALL)
    if xml_matches:
        return xml_matches[-1].strip()
    
    # Thử lấy từ \boxed{} sau đó
    boxed_matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()
        
    return text.strip()


def verify_answer(predicted: str, ground_truth: str) -> bool:
    pred_ans = extract_boxed(predicted).replace(" ", "").lower()
    true_ans = extract_boxed(ground_truth).replace(" ", "").lower()
    return pred_ans == true_ans


def format_prompt(question: str) -> str:
    system_prompt = "You are a helpful assistant. Provide reasoning enclosed in <think> tags and the final answer enclosed in <answer> tags."
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main():
    args = parse_args()
    
    # Load dataset
    print(f"[INFO] Tải dataset từ {args.test_parquet}...")
    try:
        from datasets import load_dataset
        ds = load_dataset('parquet', data_files=args.test_parquet, split='train')
    except Exception as e:
        import pandas as pd
        df = pd.read_parquet(args.test_parquet)
        ds = df.to_dict('records')
    
    # Initialize vLLM with LoRA
    print(f"\n[INFO] Đang khởi tạo vLLM engine ({args.model_name})...")
    llm = LLM(
        model=args.model_name,
        enable_lora=True,
        max_lora_rank=64, # Dự phòng dư dả
        tensor_parallel_size=1,
        gpu_memory_utilization=0.45,
        max_model_len=4096,
        trust_remote_code=True,
    )
    
    lora_request = None
    if os.path.exists(args.lora_dir) and os.listdir(args.lora_dir):
        print(f"[INFO] Load LoRA adapter từ: {args.lora_dir}")
        lora_request = LoRARequest("amsb_qa", 1, args.lora_dir)
    else:
        print("[WARN] Thư mục LoRA trống. Chạy test trên Base model.")
        
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[151645], # <|im_end|>
    )
    
    # Group prompts by source_dataset
    dataset_groups = {}
    for item in ds:
        src = item.get('source_dataset', 'unknown')
        if src not in dataset_groups:
            dataset_groups[src] = []
        dataset_groups[src].append(item)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n[INFO] Bắt đầu quá trình Generate...")
    
    for dataset_name, items in dataset_groups.items():
        print(f"  → Xử lý bộ dữ liệu: {dataset_name} ({len(items)} câu hỏi)...")
        
        prompts = [format_prompt(item['prompt']) for item in items]
        
        # Batch Generate
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora_request
        )
        
        # Export to JSONL format
        out_file = Path(args.output_dir) / f"{dataset_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for i, output in enumerate(outputs):
                gt = str(items[i]['answer'])
                question = items[i]['prompt']
                qid = items[i].get('source_id', i)
                
                for gen_id, vllm_out in enumerate(output.outputs):
                    pred_text = vllm_out.text
                    pred_ans = extract_boxed(pred_text)
                    is_correct = verify_answer(pred_text, gt)
                    
                    row = {
                        "dataset": dataset_name,
                        "question_id": qid,
                        "generation_id": gen_id,
                        "prompt": question,
                        "question": question,
                        "answer": gt,
                        "gold_answer": gt,
                        "response": pred_text,
                        "pred_answer": pred_ans,
                        "label": is_correct,
                        "method": "amsb-latest-checkpoint",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
        print(f"  → Đã lưu {len(items) * args.n_samples} kết quả dự đoán vào {out_file}")

    print("\n[SUCCESS] Hoàn tất quá trình sinh Inference Test!")

if __name__ == "__main__":
    main()
