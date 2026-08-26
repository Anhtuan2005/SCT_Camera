# Changelog — Paper_48 frozen single-participant pilot

- **Tóm tắt:** thay các tuyên bố Risk chưa được chứng minh bằng phạm vi pilot 20 clip, split 10 development + 10 frozen test, tổng 1.344,240 giây; nêu cả kết quả đúng, bỏ sót, báo giả và giới hạn một người tham gia.
- **Mục 1.2 — Đóng góp:** sửa semantics intrusion/line crossing theo code freeze; mô tả 22-feature audit; thay tuyên bố đánh giá chung bằng pilot một người có công bố kết quả âm.
- **Mục 2.5 — Công trình liên quan:** giữ hồi quy logistic như pipeline dự kiến, không mô tả là model đã huấn luyện/xác thực.
- **Mục 3 — Kiến trúc:** làm rõ `gate_alerts=false` trong pilot và kết quả hành vi đến trực tiếp từ rule runtime.
- **Mục 3.5 — Phân tích hành vi:** sửa intrusion thành crossing IN, làm rõ counter IN/OUT và runner instrumentation; thêm **Bảng 1** với điều kiện, threshold, trigger và dedup của intrusion, loitering, suspicious behavior, theft và line crossing; giữ toàn bộ bảng trên cùng một trang.
- **Mục 3.6 — Đặc trưng hành vi và cổng rủi ro:** bỏ sơ đồ/diễn giải có thể hiểu nhầm Risk đã được huấn luyện; thêm **Bảng 2** chứa đúng 22 feature theo code, nguồn tính và normalization; ghi rõ ngưỡng 0,65 là cấu hình tĩnh, không báo Risk Accuracy/Precision/Recall/F1/ROC-AUC.
- **Mục 3.7 — Phân phối cảnh báo:** rút gọn mô tả để giữ giới hạn trang; hình AlertManager được đánh số lại thành **Hình 3**.
- **Mục 4.1 — Môi trường và thiết lập thí nghiệm:** thêm thiết kế một người P01, hai repetition/session, daylight/low-light, frontal-/side-oblique, 666,296 giây development + 677,944 giây test và giới hạn không participant-disjoint.
- **Mục 4.2–4.4:** giữ các số đo kỹ thuật cũ; đánh số lại bảng thông lượng/độ trễ thành **Bảng 3–4** và hình tổng hợp thành **Hình 4**; ghi rõ 135 ca (130 đạt, 5 chưa đạt) là đánh giá hồi quy lịch sử, còn suite hiện tại ngày 19/08/2026 đạt **206/206 ca, 0 thất bại**.
- **Mục 4.5 — Pilot hành vi:** thay case theft cũ bằng giao thức frozen-test; chèn **Hình 5 — “Pilot behavior evaluation outcomes on the frozen test split”** và **Bảng 5** với TP/FP/TN/FN, DR/FAR, Precision/Recall/F1 và latency cho từng behavior; không tính aggregate score; bỏ title dư bên trong Hình 5 vì caption đã nằm dưới hình.
- **Mục 5 — Thảo luận và hạn chế:** bổ sung năm intrusion event trùng và diễn đạt chính xác là “cơ chế loại bỏ sự kiện trùng lặp theo track ID chưa ổn định”; ghi nhận loitering/suspicious test thành công, theft miss + false alert, line-crossing event sai/trùng và bằng chứng thời gian cụ thể; nêu rõ không có participant-level generalization và không có Risk metrics hợp lệ.
- **Mục 6 — Kết luận:** đồng bộ mọi claim với pilot đóng băng và xác định bước tiếp theo là participant-disjoint validation sau khi sửa lỗi quan sát được.
- **Bố cục:** giữ nguyên style, khổ trang và hệ thống đánh số của bản nguồn; bản cuối render đủ **10 trang**, không có bảng/hình bị cắt hoặc chồng lấn. Tệp nguồn `E:\Paper_48.docx` không bị sửa.
