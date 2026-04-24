# Hướng Dẫn Chạy M-SB-GRPO Benchmark trên Server H100

Tài liệu này hướng dẫn cách khởi chạy và đánh giá (benchmark) toàn bộ các mô hình open-r1 (Vanilla, MGRPO, SEED, A-MSB) trên server Linux được trang bị GPU H100, đảm bảo an toàn dữ liệu khi có sự cố.

## 1. Cơ chế an toàn (Crash-proof Checkpointing)

Mặc định, các file config đã được cập nhật để lưu checkpoint dọc đường thay vì chỉ lưu ở cuối epoch. Cụ thể:
- `save_strategy`: "steps"
- `save_steps`: 50
- `save_total_limit`: 2

**Lợi ích:** 
Nếu server bị tắt đột ngột (hết giờ, sập nguồn), lần chạy tiếp theo script sẽ tự động quét thư mục output (`data/Qwen2.5-Math-7B-Open-R1-*`), phát hiện checkpoint gần nhất (vd: `checkpoint-150`) và **tự động resume** quá trình train từ điểm đó mà không bị mất dữ liệu.

## 2. Các bước thực thi trên Server

### Bước 2.1. Chuẩn bị (Clone & Navigate)
Đảm bảo bạn đang ở thư mục gốc của dự án trên server:
```bash
cd /path/to/M-SB-GRPO
```

Nếu bạn có sử dụng Weights & Biases (WandB) để theo dõi quá trình train, hãy tạo file `.env` ở root:
```bash
echo "WANDB_API_KEY=your_api_key_here" > .env
```
*(Nếu không dùng WandB, bạn có thể truyền thêm cờ `--disable-wandb` ở Bước 2.2)*

### Bước 2.2. Khởi chạy All-in-One Pipeline
Để chạy toàn bộ từ setup môi trường, kéo data, kiểm tra cấu hình, train và đánh giá cả 4 phương pháp, chạy lệnh:

```bash
cd compare/open-r1
chmod +x ./scripts/run_all_h100_full.sh

# Chạy toàn bộ luồng
./scripts/run_all_h100_full.sh
```

**Các cờ (flags) hữu ích nếu bạn chỉ muốn chạy một phần:**
- Chạy phương pháp cụ thể: `./scripts/run_all_h100_full.sh --methods vanilla,amsb`
- Bỏ qua bước train (nếu chỉ muốn benchmark): `./scripts/run_all_h100_full.sh --skip-train`
- Tắt WandB: `./scripts/run_all_h100_full.sh --disable-wandb`
- Tiếp tục nếu có 1 method bị lỗi: `./scripts/run_all_h100_full.sh --continue-on-error`

Mọi logs trong quá trình chạy sẽ được lưu cẩn thận vào `compare/open-r1/logs/pipeline-h100/`.

## 3. Tổng hợp kết quả và vẽ biểu đồ

Sau khi script trên chạy xong (hoàn tất cả train và eval cho các phương pháp), bạn cần tổng hợp số liệu để ra báo cáo cuối cùng.

Quay lại thư mục root và kích hoạt môi trường ảo:
```bash
cd ../.. 
source compare/open-r1/.venv/bin/activate
```

Chạy orchestrator để tổng hợp số liệu:
```bash
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage collect
```

Vẽ biểu đồ so sánh:
```bash
python compare/run_compare_pipeline.py --config compare/pipeline.compare.yaml --stage chart --metric pass@1
```

**Kết quả thu được:**
Bạn sẽ tìm thấy kết quả tổng hợp trong thư mục `compare/artifacts/compare/`:
- `normalized_metrics.csv` và `.json`: Chứa số liệu pass@1, pass@2,... thô.
- `chart_overall_pass@1.png` và `chart_by_dataset_pass@1.png`: Biểu đồ so sánh trực quan.
