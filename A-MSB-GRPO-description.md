# ĐẶC TẢ KIẾN TRÚC HỆ THỐNG: A-MSB-GRPO

**(Adaptive Multi-Layer Semantic-Balanced GRPO)**

**Tóm tắt kiến trúc (Architecture Abstract):** A-MSB-GRPO là một kiến trúc học tăng cường (Reinforcement Learning) đa tầng dành cho Mô hình Ngôn ngữ Lớn (LLM). Kiến trúc này được tinh chỉnh nhằm giải quyết ba điểm nghẽn kỹ thuật cốt lõi của thuật toán GRPO truyền thống: (1) Hiện tượng suy biến Gradient (Advantage Vanishing) khi batch dữ liệu đồng nhất; (2) Thiên kiến sửa lỗi (Correction Bias) trong quá trình mô hình tự suy luận; và (3) Hiện tượng phân mảnh bộ nhớ (Memory Fragmentation) do kích thước batch động. Hệ thống vận hành dựa trên cơ chế định tuyến động (Dynamic Routing) và tối ưu hóa bằng các đại lượng toán học liên tục, loại bỏ hoàn toàn các lệnh rẽ nhánh rời rạc gây giảm hiệu suất phần cứng.

------

### GIAI ĐOẠN 1: KHỞI TẠO VÀ ỔN ĐỊNH CƠ SỞ (BASE GENERATION & STABILIZATION)

**Bước 1.1: Khởi tạo mẫu tốc độ cao (High-Throughput Rollout)**

- Với một truy vấn đầu vào (Prompt), mô hình tại Layer 1 tiến hành sinh song song $N$ mẫu kết quả độc lập (khuyến nghị $N=8$). Quá trình này tối ưu hóa thông lượng (throughput) thông qua cơ chế vLLM, không áp dụng các bước phân tích ngữ nghĩa phức tạp.
- Hệ thống đánh giá (Rule-based Verifier) xác thực $N$ kết quả để tính toán Tỷ lệ Chính xác (Correct Ratio) của toàn bộ Batch.

**Bước 1.2: Định tuyến động (Conditioning Gate)**

Nhằm tối ưu hóa tài nguyên tính toán (Compute Efficiency), hệ thống định tuyến các mẫu dữ liệu dựa trên Tỷ lệ Chính xác:

- **Nhánh 1 (Trường hợp cực đoan - 0% hoặc 100%):** Nếu mô hình giải quyết sai toàn bộ hoặc chính xác toàn bộ, quá trình chuyển tiếp sang Layer 2 bị hủy bỏ để tiết kiệm tài nguyên. Hệ thống áp dụng thuật toán **NGRPO (Negative-enhanced GRPO)** trực tiếp tại Layer 1. Cơ chế *Advantage Calibration* tích hợp phần thưởng ảo (Virtual Reward) giúp duy trì giá trị Advantage âm đối với batch 0%, và tự động triệt tiêu ở batch 100%, đảm bảo không gian hàm Loss liên tục và không bị gián đoạn đạo hàm.
- **Nhánh 2 (Batch hỗn hợp - 0% < Ratio < 100%):** Toàn bộ Batch dữ liệu (bao gồm cả mẫu đúng và mẫu sai) được chuyển tiếp sang Giai đoạn 2. Việc truyền toàn bộ Batch giúp khắc phục triệt để hiện tượng "Correction Bias", ngăn mô hình hình thành phản xạ thay đổi đáp án một cách máy móc khi bước vào tập huấn luyện tinh chỉnh.

------

### GIAI ĐOẠN 2: TỰ ĐÁNH GIÁ VÀ HIỆU CHỈNH CHUYÊN SÂU (BIAS-FREE SELF-REFLECTION & CORRECTION)

**Bước 2.1: Lấy mẫu tăng cường (Augmented Sampling)**

- Các kết quả từ Layer 1 được đưa vào Layer 2 kèm theo một System Prompt mang tính trung lập: *"Hãy kiểm tra lại tính logic của các bước giải. Giữ nguyên nếu phát hiện đáp án đã chính xác, hiệu chỉnh lại nếu phát hiện lỗi sai"*.
- Hệ thống thực thi Augmented Sampling: Mỗi mẫu dữ liệu gốc được yêu cầu sinh ra $M$ nhánh suy luận/hiệu chỉnh mới (khuyến nghị $M=4$ hoặc $8$), tạo ra một không gian mẫu mở rộng ($N \times M$ mẫu).

**Bước 2.2: Phân tách không gian dữ liệu (Evaluate & Pool Splitting)**

- Trình đánh giá tiến hành nghiệm thu tập dữ liệu mở rộng.
- Dữ liệu được phân tách thành hai tập hợp độc lập: **Correct Pool** (Tập hợp các mẫu có đáp án đúng) và **Error Pool** (Tập hợp các mẫu có đáp án sai).

**Bước 2.3: Phân tích cụm lỗi ngữ nghĩa (Semantic Error Profiling)**

- Dữ liệu từ *Error Pool* được chuyển đổi thành các biểu diễn vector (Embeddings) thông qua một mô hình nhúng tối ưu hóa (Lightweight Embedder, ví dụ: `all-MiniLM-L6-v2`) chạy trên CPU để giảm tải cho GPU.
- Áp dụng thuật toán phân cụm **K-Means Clustering** trên không gian vector để nhóm các mẫu sai có độ tương đồng cao về bản chất logic.
- Hệ thống trích xuất **Top Error Cluster** (Cụm lỗi chiếm tỷ trọng lớn nhất). Đây được xác định là lỗ hổng suy luận hệ thống cốt lõi mà mô hình cần ưu tiên khắc phục.

------

### GIAI ĐOẠN 3: CẤU TRÚC TENSOR VÀ TỐI ƯU HÓA CHÍNH SÁCH (TENSOR SHAPING & POLICY UPDATE)

**Bước 3.1: Xây dựng Batch tĩnh cân bằng (Static Balanced Batch Construction)**

Để đạt hiệu suất tính toán cực đại trên GPU (tối đa hóa TFLOPS) và đáp ứng định lý tối ưu phương sai của kiến trúc DIVA-GRPO, hệ thống xây dựng một Tensor huấn luyện với kích thước tĩnh (Static Size $K$, ví dụ $K=16$) theo tỷ lệ phân bổ $50/50$:

- Lấy $K/2$ mẫu từ *Correct Pool*.
- Lấy $K/2$ mẫu từ *Top Error Cluster*.
- **Cơ chế kỹ thuật cốt lõi:** Hệ thống sử dụng thuật toán **Lấy mẫu có hoàn lại (Sampling with Replacement)**. Bất chấp sự chênh lệch kích thước của các tập dữ liệu thành phần, thuật toán tự động nội suy và lấp đầy khuôn Tensor tĩnh. Quá trình này loại bỏ hoàn toàn các tác vụ biến đổi kích thước động (Dynamic reshaping) hoặc thêm đệm (Padding), giúp ngăn chặn phân mảnh VRAM.

**Bước 3.2: Điều chỉnh biên độ cập nhật liên tục (Continuous Loss Scaling via SEED Entropy)**

Hệ thống sử dụng lý thuyết thông tin để tự động điều chỉnh cường độ cập nhật trọng số:

- Tính toán **Shannon Entropy ($H$)** của phân phối các cụm lỗi trong *Error Pool* nhằm đo lường độ bất định (Uncertainty) trong suy luận của mô hình.
- Chuyển đổi $H$ thành hệ số tỷ lệ nghịch (Ví dụ: $Scale = e^{-H}$).
- Hệ số này được nhân trực tiếp vào hàm Loss của thuật toán GRPO:
  - Khi hệ thống vấp phải lỗi logic cố hữu (Entropy thấp) $\rightarrow$ Hệ số Scale đạt mức tối đa, tối ưu hóa cường độ gradient để triệt tiêu lỗi.
  - Khi mô hình xuất hiện các lỗi ngẫu nhiên/nhiễu (Entropy cao) $\rightarrow$ Hệ số Scale tự động giảm tiệm cận 0, hạ thấp biên độ cập nhật để bảo vệ mô hình khỏi hiện tượng sụp đổ trọng số (Gradient Whiplash).

**Bước 3.3: Lan truyền ngược (Backpropagation & Update)**

Thực thi truyền ngược hàm Loss để cập nhật Chính sách (Policy). Kiến trúc hỗ trợ tích hợp với LoRA/QLoRA và các framework tối ưu hóa bộ nhớ tĩnh như DeepSpeed (ZeRO-2/ZeRO-3) để đảm bảo tiến trình huấn luyện vận hành ổn định trên các nền tảng phần cứng giới hạn (như Node đơn H100 80GB).