# FINAL REPORT AUDIT

- Ngày hoàn tất: 2026-08-20
- File nguồn: `E:\SCT_Camera\outputs\report_fall_update_20260820\BAOCAO_TichHop_Pilot_CapNhatFall.docx`
- File final: `E:\SCT_Camera\outputs\final_report_20260820\BAOCAO_FINAL.docx`
- Số trang sau khi cập nhật field và dàn trang bằng Microsoft Word: 116 trang.

## 1. Caption bảng đã sửa

Đã chuẩn hóa 12 caption, giữ nguyên nội dung và số liệu trong bảng:

1. Bảng 2-1: So sánh các phiên bản YOLO hiện đại (v8 – YOLO26) trên các tiêu chí quyết định.
2. Bảng 2-2: Ma trận IoU giữa các track hiện có và các detection mới.
3. Bảng 2-3: So sánh kết quả gán ghép giữa chiến lược Greedy và thuật toán Hungarian.
4. Bảng 2-4: Tóm tắt các độ đo đánh giá hệ thống MOT.
5. Bảng 2-5: Các vấn đề thực tế thường gặp và cách xử lý.
6. Bảng 2-6: Các chỉ số đánh giá chất lượng theo dõi đa đối tượng.
7. Bảng 4-1: Thông số hai môi trường thực nghiệm.
8. Bảng 4-2: AI FPS tổng hợp theo số lượng camera trong phép đo 10 phút.
9. Bảng 4-3: Độ trễ trung bình từng giai đoạn xử lý AI (trường hợp một camera).
10. Bảng 4-4: Tổng hợp các nhóm testcase tự động theo thành phần hệ thống.
11. Bảng 4-5: Kết quả từng hành vi trên tập kiểm thử đóng băng.
12. Bảng 4-6: Kết quả kiểm thử độ ổn định bốn camera trong 8 giờ.

## 2. Automated tests

Kết quả cuối cùng thuộc trạng thái source code cục bộ ngày 2026-08-20:

- Lệnh xác nhận toàn bộ suite: `python -m pytest -q`
- Kết quả: **206 passed, 3 warnings in 17.08s**.
- Lệnh xác nhận nhóm fall detection: `python -m pytest -q tests/test_fall_detection.py tests/test_person_alerts.py tests/test_pipeline.py`
- Kết quả nhóm fall: **72 passed in 0.46s**; đây là tập con của 206 testcase, không phải một mốc version riêng để cộng thêm.
- Git HEAD tại thời điểm audit: `a7881f788252a1b6c681430f99d7d284719a8ee6`.
- Working tree có thay đổi cục bộ; vì vậy con số 206 mô tả đúng trạng thái source đang dùng để hoàn thiện báo cáo ngày 2026-08-20, không chỉ riêng clean commit trên.

Phân bố 206 testcase theo 21 file:

| File | Số test |
|---|---:|
| test_asset_watch.py | 6 |
| test_auth.py | 5 |
| test_database.py | 3 |
| test_detector.py | 6 |
| test_drawing.py | 15 |
| test_fall_detection.py | 24 |
| test_frame_buffer.py | 2 |
| test_intrusion.py | 1 |
| test_line_counter.py | 5 |
| test_monitor_runtime.py | 2 |
| test_person_alerts.py | 29 |
| test_person_identity.py | 17 |
| test_pipeline.py | 20 |
| test_pose.py | 2 |
| test_pose_classifier.py | 13 |
| test_priority_behavior_experiment.py | 6 |
| test_revision_experiments.py | 6 |
| test_runtime_settings.py | 11 |
| test_theft_behavior.py | 3 |
| test_tracker_cmc.py | 23 |
| test_web_responsiveness.py | 7 |
| **Tổng** | **206** |

Đã cập nhật nhất quán con số 206 tại mục 4.4, phần mở đầu 4.5, mục 4.8, mục 5.1, mục 5.2 và các vị trí liên quan; kết quả cũ 175 tests không còn được trộn với revision hiện tại.

## 3. Nội dung và consistency đã cập nhật

- Mục 1.6 đã mô tả đủ scaling, độ trễ, automated tests, 7 E2E, pilot năm hành vi trên frozen test, hai phiên stability 4 camera × 8 giờ, kết quả và hạn chế.
- Đã bổ sung câu phân biệt mục 4.5 (E2E functional/business testing) và 4.6 (quantitative frozen pilot), tránh mâu thuẫn giữa kết quả PASSED theo kịch bản với failure modes của pilot.
- Chương 5 tiếp tục nêu rõ giới hạn của Risk, face threshold, pilot một participant, failure modes của intrusion/theft/line crossing và phạm vi hai GPU, tối đa bốn camera.
- Giữ nguyên toàn bộ số liệu scaling RTX 3050/RTX 5060 Ti, hai phiên stability 8 giờ, pilot 20 clip và các metric frozen test.
- Không thêm ROC/EER/FAR/FRR, không claim Risk đã được quantitative validation và không suy rộng kết quả sang mọi GPU, hơn bốn camera hoặc RTSP.

## 4. Mục lục, field và dàn trang

- Đã sửa số trang La Mã của front matter theo chuỗi liên tục; Lời cảm ơn bắt đầu tại i, Lời cam đoan tại ii và Mục lục tại iii.
- Đã cập nhật Table of Contents, Danh mục bảng, Danh mục hình và các field/cross-reference bằng Microsoft Word.
- Đã loại bỏ 7 ngắt trang dư trong danh sách testcase và 2 dòng trống làm phát sinh một trang trắng trước Chương 3.
- Đã giữ các dòng mục lục/danh mục không bị tách trang, giữ hàng tổng kết của bảng trên cùng một trang và bỏ ngắt trang không cần thiết trước mục 4.5.
- Không thay đổi font, margin, line spacing hoặc template tổng thể.

## 5. Kết quả QA cuối

- DOCX mở hợp lệ: PASS.
- 38 bảng được bảo toàn: PASS.
- 35 hình/đối tượng ảnh và 35 drawing reference được bảo toàn: PASS.
- Caption bảng và Danh mục bảng khớp chính xác: PASS.
- Không còn caption lỗi dạng `Bảng KẾT QUẢ THỰC NGHIỆM-...` hoặc `Bảng CƠ SỞ LÝ THUYẾT-...`: PASS.
- Không có lỗi field dạng `Error! Reference source not found`: PASS.
- Đã render và kiểm tra trực quan toàn bộ 116 trang; không còn trang trắng không cần thiết, heading mồ côi hoặc bảng/hình bị cắt do các chỉnh sửa lần này.

## 6. Inconsistency còn tồn tại

Không còn inconsistency trọng yếu trong phạm vi yêu cầu. Các giới hạn thực nghiệm vẫn được giữ nguyên và trình bày công khai: hai GPU, tối đa bốn nguồn video cục bộ, pilot nhỏ không participant-disjoint, face threshold chưa có ROC/EER/FAR/FRR độc lập, Risk chưa được xác thực định lượng và hai phiên 8 giờ không thay thế kiểm thử dài hơn.
