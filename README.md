# 🔬 M-SB-GRPO (Adaptive Multi-Layer Semantic-Balanced GRPO)

Dự án này là một hệ thống huấn luyện Reinforcement Learning (RL) tiên tiến dành cho Mô hình Ngôn ngữ Lớn (LLM), tập trung vào khả năng tự sửa lỗi (Self-Correction) trong các tác vụ suy luận toán học. Dự án hiện thực hóa kiến trúc **M-SB-GRPO** (còn gọi là A-MSB-GRPO), một giải pháp đột phá khắc phục 3 nhược điểm lớn của GRPO truyền thống: Suy biến Gradient (Advantage Vanishing), Thiên kiến sửa lỗi (Correction Bias), và Phân mảnh bộ nhớ (Memory Fragmentation).

---

## 🏛️ 1. Kiến trúc M-SB-GRPO cốt lõi

Kiến trúc **Adaptive Multi-Layer Semantic-Balanced GRPO** là sự kết hợp hoàn hảo các lý thuyết tối ưu hóa từ 3 bài báo nền tảng: *Multi-Layer GRPO*, *SEED-GRPO*, và định lý toán học 50/50 từ *DIVA-GRPO*. Hệ thống hoạt động qua 3 giai đoạn:

### Giai đoạn 1: Khởi tạo và Ổn định (Base Generation)
- **High-Throughput Rollout:** Tại Layer 1, mô hình sinh song song $N$ mẫu kết quả độc lập qua engine vLLM. 
- **NGRPO (Negative-enhanced GRPO):** Nếu Batch cực đoan (0% đúng hoặc 100% đúng), quá trình chuyển tiếp sang Layer 2 bị hủy. Hệ thống kích hoạt thuật toán NGRPO với phần thưởng ảo (Virtual Reward) để hiệu chuẩn Advantage, duy trì không gian hàm Loss liên tục.
- **Chống Correction Bias:** Nếu Batch có kết quả hỗn hợp, **toàn bộ** Batch được chuyển sang Layer 2 để tránh việc mô hình hình thành phản xạ thay đổi đáp án một cách máy móc.

### Giai đoạn 2: Tự đánh giá và Gom cụm lỗi (Self-Reflection & Profiling)
- **Augmented Sampling:** Mỗi mẫu ở Layer 1 sinh ra $M$ nhánh hiệu chỉnh với System Prompt yêu cầu tự kiểm tra lại logic.
- **Semantic Error Profiling (Học thuyết SEED):** Các mẫu có đáp án sai (Error Pool) được nhúng (embed) thành vector bằng mô hình `all-MiniLM-L6-v2` (chạy trên CPU) và gom cụm bằng **K-Means Clustering**. 
- Hệ thống trích xuất **Top Error Cluster** (Cụm lỗi chiếm tỷ trọng lớn nhất) – đại diện cho "lỗi hệ thống" nguy hiểm nhất mà mô hình đang tin là đúng.

### Giai đoạn 3: Tensor tĩnh và Tối ưu hóa (Tensor Shaping & Policy Update)
- **Static Balanced Batching (Học thuyết DIVA):** Hệ thống dùng kỹ thuật "Lấy mẫu có hoàn lại" (Sampling with Replacement) để ép khuôn một Tensor huấn luyện kích thước tĩnh $K$ (ví dụ: $K=16$) với **tỷ lệ vàng 50/50**: 50% từ Correct Pool và 50% từ Top Error Cluster. Việc này tối đa hóa độ lớn của Gradient Update (theo Theorem C.3 của DIVA-GRPO) và chống phân mảnh VRAM.
- **Continuous Loss Scaling:** Tính Shannon Entropy ($H$) của phân phối các cụm lỗi. Hệ số $Scale = e^{-H}$ được nhân trực tiếp vào hàm Loss. Khi Entropy thấp (mô hình mắc lỗi hệ thống), hàm Loss được tối đa hóa để "đấm" mạnh vào lỗi sai. Khi Entropy cao (mô hình đoán mò ngẫu nhiên), hàm Loss bị triệt tiêu để chống sụp đổ trọng số.

---

## 📊 2. Các Baselines So sánh

Để chứng minh sự vượt trội của phương pháp chủ động "xây dựng batch cân bằng ngữ nghĩa" trong M-SB-GRPO, dự án đối chiếu trực tiếp với 3 Baselines:

1. **Vanilla GRPO:** Kiến trúc nền tảng 1 lớp (theo DeepSeekMath). Không có cơ chế tự sửa lỗi và không cân bằng. Được dùng làm điểm khởi đầu.
2. **MGRPO (Multi-Layer GRPO):** Kiến trúc 2 lớp. Layer 2 lấy mẫu **ngẫu nhiên** để sửa lỗi (không có gom cụm, không cân bằng). Baseline này dùng để chứng minh sự ưu việt của "cân bằng batch thông minh" so với "sửa lỗi ngẫu nhiên".
3. **SEED-GRPO:** Kiến trúc 1 lớp, dùng Semantic Entropy để giảm trọng số advantage một cách **thụ động**. Baseline này dùng để so sánh giữa việc "chủ động tạo môi trường học" (M-SB-GRPO) so với "thụ động điều chỉnh trọng số".

---

## 📚 3. Tập dữ liệu & Benchmarks (Datasets)

- **Huấn luyện:** Tập tin hợp đồng dữ liệu `combined_train.parquet` chứa các problem và lời giải chuẩn.
- **Đánh giá (Benchmarks):** Kiểm thử trên các bộ dữ liệu Toán học khắt khe nhất: **MATH-500**, **GSM8K**, **AIME 2024/2025**, và **OlympiadBench**.
- **Metrics:** Thu thập tự động các chỉ số Pass@1, Pass@2, Pass@4, Pass@8.

---

## ⚙️ 4. Chi tiết Implement (Nền tảng H100 Server)

Mọi pipeline đều được ép chạy chung một bộ cấu hình nghiêm ngặt để đảm bảo sự công bằng tuyệt đối trong thực nghiệm:

- **Phần cứng:** 1x GPU NVIDIA H100 (80GB VRAM).
- **Mô hình nền:** `Qwen/Qwen2.5-7B-Instruct`.
- **Tối ưu hóa tài nguyên:** 
  - **4-bit QLoRA** (`use_peft=True`, `load_in_4bit=True`).
  - Tích hợp **vLLM** làm generation engine (Cấp phát 30% VRAM GPU).
  - Tích hợp DeepSpeed ZeRO-3 và Flash Attention 2.
- **Siêu tham số (Fair Comparison):** 
  - Learning Rate: 1.0e-05.
  - Prompt Length = 2048, Completion = 1024.
  - Generative Rollouts: 8 câu trả lời/prompt.

---

## 🚀 5. Hướng dẫn sử dụng nhanh

Pipeline được điều phối 100% tự động qua hệ thống Bash script.

**Khởi chạy Huấn luyện cho Baselines (Vanilla, MGRPO, SEED):**
```bash
cd compare/open-r1
./scripts/run_baselines_h100.sh
```

**Tổng hợp và Visualize Kết quả:**
```bash
# Tổng hợp các bảng summary
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage collect

# Trích xuất biểu đồ tự động
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage chart --metric pass@1
```
