**Vai trò (Role):** Hãy đóng vai trò là một Kỹ sư Hệ thống Trí tuệ Nhân tạo Cao cấp (Senior AI Systems Engineer), chuyên gia về Huấn luyện Mô hình Ngôn ngữ Lớn (LLM) và Tối ưu hóa Học tăng cường dựa trên phản hồi (RLHF/GRPO).

**Nhiệm vụ (Task):** Yêu cầu hiện tại là triển khai mã nguồn thực chiến (production-ready code) cho kiến trúc thuật toán **A-MSB-GRPO** nhằm mục đích huấn luyện mô hình **[Qwen-2.5-7B-Instruct]** giải quyết các bài toán định lý và logic (Tập dữ liệu MATH benchmark).

**1. Ràng buộc Hệ thống và Nền tảng Công nghệ (Hardware & Framework Constraints):**

- **Phần cứng:** 1x GPU NVIDIA H100 (80GB VRAM). Tiêu chí tối ưu hóa bộ nhớ VRAM là ưu tiên tuyệt đối.
- **Thư viện lõi:** PyTorch, HuggingFace `transformers`.
- **Tối ưu hóa tài nguyên:** Bắt buộc tích hợp thư viện `vLLM` cho các tác vụ sinh văn bản (Generation) ở tốc độ cao; sử dụng kỹ thuật LoRA/QLoRA kết hợp với DeepSpeed (ZeRO-2 hoặc ZeRO-3) để quản lý trạng thái Optimizer.

**2. Đặc tả Kiến trúc Thuật toán (Architecture Specification):**

Dưới đây là bản thiết kế hệ thống chi tiết. Vui lòng phân tích kỹ luồng dữ liệu (Data Flow), cơ chế định tuyến (Routing) và phương pháp tính toán hàm Suy hao (Loss Function):

Đọc file `A-MSB-GRPO-description.md`

**3. Yêu cầu Nghiêm ngặt về Đầu ra (Strict Output Requirements):**

Tuyệt đối không sử dụng mã giả (pseudo-code) hoặc các hàm giữ chỗ (dummy functions) chung chung. Yêu cầu triển khai chi tiết mã nguồn bằng ngôn ngữ Python/PyTorch cho 3 module cốt lõi sau:

- **Module 1: Custom A-MSB-GRPO Loss Function.** Triển khai hàm tính Loss tích hợp thuật toán NGRPO (hiệu chuẩn bằng Virtual Reward) và cơ chế điều chỉnh biên độ cập nhật (Continuous Loss Scaling) sử dụng tham số Shannon Entropy. Cần đảm bảo các phép toán Tensor không làm gián đoạn luồng Gradient.
- **Module 2: Dữ liệu và Lấy mẫu (Data Shaping & Sampling).** Viết hàm tiếp nhận `correct_pool` và `top_error_cluster`, thực thi thuật toán Lấy mẫu có hoàn lại (Sampling with Replacement) để trả về một Tensor Batch có kích thước tĩnh (Static Batch Size $K=16$), đảm bảo tỷ lệ 50/50.
- **Module 3: Vòng lặp Huấn luyện (Generation & Training Loop).** Trình bày cấu trúc vòng lặp chính, xử lý việc truyền nhận dữ liệu giữa engine sinh văn bản (`vLLM`) và engine tính toán đạo hàm (`PyTorch`) mà không gây hiện tượng tràn bộ nhớ (Out of Memory).

**Hướng dẫn thực thi:** Vui lòng bắt đầu bằng việc lập trình và giải thích **Module 1 (Custom Loss Function)**. Trong từng đoạn mã, hãy đính kèm các comment (chú thích) giải thích rõ sự thay đổi về kích thước (shape) của Tensor qua từng phép biến đổi.