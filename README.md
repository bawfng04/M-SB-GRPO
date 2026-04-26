# 🔬 A-MSB-GRPO: Adaptive Multi-Stage Branching Group Relative Policy Optimization

Dự án này là một hệ thống (pipeline) huấn luyện Reinforcement Learning (RL) trên Large Language Models (LLMs) dành cho các tác vụ suy luận toán học (Mathematical Reasoning). Điểm nhấn của dự án là việc đề xuất và triển khai kiến trúc **A-MSB-GRPO** cùng với việc xây dựng một hệ thống đánh giá tự động (benchmarking orchestrator) để so sánh phương pháp mới này với các biến thể GRPO State-of-the-Art khác.

---

## 🏛️ 1. Kiến trúc A-MSB-GRPO (Đề xuất)

A-MSB (Adaptive Multi-Stage Branching) mở rộng thuật toán GRPO truyền thống bằng cách kết hợp chiến lược sinh mẫu đa tầng (multi-stage) và tối ưu hóa dựa trên sự đa dạng lỗi sai (error profiling).

**Cấu trúc thuật toán cốt lõi:**
1. **Layer 1 (Base Rollouts):** Sinh ra `N_ROLLOUTS` (ví dụ: 8) câu trả lời ban đầu cho mỗi prompt toán học.
2. **K-Means Error Profiling:** Thay vì chỉ đánh giá đúng/sai (1/0), hệ thống trích xuất đặc trưng của các câu trả lời sai và gom cụm (Clustering) thành `N_CLUSTERS=4` cụm lỗi (Error Clusters) để tìm ra các kiểu suy luận sai lầm phổ biến nhất.
3. **Layer 2 (Self-Reflection Branches):** Dựa trên các cụm lỗi, hệ thống sinh ra `M_REFLECTIONS=2` nhánh phản tư (reflection branches) ép model tự đánh giá lại các lỗi logic của mình. (Sử dụng prompt: *"Review the reasoning carefully. Keep the answer unchanged if it is already correct, otherwise provide a corrected solution."*).
4. **Adaptive Entropy Scaling:** Tính toán Semantic Entropy (độ hỗn loạn ngữ nghĩa) của các câu trả lời. Nếu entropy thấp (model rất tự tin nhưng lại sai), hệ thống phạt nặng hơn thông qua hàm `exp_decay`.
5. **Static Balanced Batching:** Duy trì một batch tĩnh tỷ lệ 50/50 (4 đúng, 4 sai) để làm mượt gradient trong quá trình PPO-style clipping.

---

## 📊 2. Các Baselines So sánh

Để chứng minh tính hiệu quả của A-MSB-GRPO, hệ thống triển khai pipeline so sánh trực tiếp với 3 phương pháp baselines:

1. **Vanilla GRPO (Chuẩn mực):** GRPO cơ bản, chỉ sinh text 1 lần và chấm điểm (Accuracy, Format, Tag Count) theo phương pháp PPO đơn giản (tham chiếu theo bài báo DeepSeekMath).
2. **MGRPO (Multi-Generation):** Sử dụng các Guiding Phrases để điều hướng model tự sửa sai (ví dụ: *"Re-check the final answer carefully..."*). So với A-MSB, MGRPO sinh text nhiều lớp nhưng không có cơ chế phân loại lỗi sai hay gom cụm.
3. **SEED (Semantic Entropy-driven):** Kỹ thuật điều biến hàm mất mát dựa trên Entropy ngữ nghĩa. Nó đánh trọng số các mẫu (modulation_factor) dựa trên độ phân tán của câu trả lời, nhưng thiếu cơ chế phản tư nhiều nhánh (Branching) như A-MSB.

---

## 📚 3. Datasets (Dữ liệu Huấn luyện & Đánh giá)

Hệ thống được thiết kế theo chuẩn Hợp đồng dữ liệu (Dataset Contract) `.parquet` để đồng bộ toàn bộ pipeline:

**Dữ liệu Huấn luyện (Train):**
- **Nguồn:** `data/datasets/requested/train/combined_train.parquet`.
- **Cột:** Bắt buộc có cột `problem` (câu hỏi) và `solution` (lời giải chuẩn).

**Dữ liệu Đánh giá (Benchmark Test):**
Quá trình đánh giá được thực hiện độc lập (Offline-only dryrun) qua các tập benchmark Toán học khó nhất hiện nay:
- **MATH-500**: 500 bài toán thi học sinh giỏi các cấp độ.
- **GSM8K**: Bài toán đố cấp 1 & cấp 2.
- **AIME 2025**: Các bài toán kỳ thi Toán học Mỹ 2025 (Cực khó).
- **OlympiadBench**: Các bài toán Olympic Toán quốc tế.

Hệ thống thu thập số liệu **Pass@1, Pass@2, Pass@4, Pass@8** cho tất cả các phương pháp.

---

## ⚙️ 4. Chi tiết Kỹ thuật Implement (H100 Server)

Mọi phương pháp trong pipeline được căn chỉnh (align) cấu hình phần cứng giống nhau tuyệt đối để đảm bảo **tính công bằng** trong thực nghiệm:

- **Mô hình nền:** `Qwen/Qwen2.5-7B-Instruct`
- **Bộ máy tạo text (Generation Engine):** `vLLM` (Cấp phát 30% VRAM GPU).
- **Kỹ thuật tối ưu bộ nhớ:** 
  - Huấn luyện trên **1 GPU H100 (80GB VRAM)**.
  - Sử dụng **4-bit QLoRA** (`use_peft=True`, `load_in_4bit=True`).
  - Áp dụng **DeepSpeed ZeRO-3** (Tối ưu hóa trạng thái Optimizer).
  - Bật Flash Attention 2 (`attn_implementation=flash_attention_2`).
- **Siêu tham số chung (Hyperparameters):**
  - **Epochs:** 2 
  - **Learning Rate:** 1.0e-05 (Cosine Scheduler)
  - **Sequence Length:** Prompt max 2048, Completion max 1024.
  - **Micro Batch Size:** 2 (với Gradient Accumulation = 8 để đạt Effective Batch Size = 16).
  - **Checkpointing:** Lưu trạng thái an toàn (Crash-proof) sau mỗi 50 steps.

---

## 🚀 5. Hướng dẫn chạy nhanh (Quick Start)

Pipeline được tự động hóa 100% bằng bash scripts và Python orchestrator.

**Chạy toàn bộ các Baselines (Vanilla, MGRPO, SEED):**
```bash
cd compare/open-r1
./scripts/run_baselines_h100.sh
```

**Tổng hợp số liệu & Vẽ biểu đồ báo cáo:**
```bash
# Tổng hợp các file summary.json thành tsv/csv
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage collect

# Trích xuất biểu đồ so sánh tự động
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage chart --metric pass@1
```

*Đồ thị sẽ tự động sinh ra ở mục `compare/artifacts/compare/` sẵn sàng để nhúng vào báo cáo LaTeX/Word.*
