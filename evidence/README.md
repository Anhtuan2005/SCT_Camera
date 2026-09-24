# SCT Camera Evaluation Evidence

Các bảng CSV trong thư mục này trích từ kết quả thí nghiệm đã lưu cục bộ. Chúng cho phép đối chiếu số liệu nêu trong CV và README. Video/annotation gốc không nằm trong repo công khai, nên không thể tái lập đầy đủ chỉ từ các bảng này.

## Throughput

[`runtime_benchmark.csv`](runtime_benchmark.csv) là trung bình của 3 lần chạy cho mỗi mức 1, 2 và 4 luồng trên video local 1280 x 720. Mỗi lần đo 15 giây sau 30 frame warm-up, dùng YOLO11s + pose trên RTX 3050 Laptop GPU. `aggregate_fps_mean` là tổng thông lượng; `fps_per_camera_mean` là tổng chia cho số luồng. Với 4 luồng: 27.645 FPS tổng và 6.911 FPS mỗi luồng trung bình. Đây không phải FPS riêng của một camera RTSP thực tế.

## Behavior pilot

Pilot có 40 clip: P01 và P02 mỗi người 20 clip, chia development/test; mỗi test split có 10 clip, gồm 1 clip dương và 1 clip âm cho từng hành vi. [`p01_test_metrics.csv`](p01_test_metrics.csv) và [`p02_test_metrics.csv`](p02_test_metrics.csv) là hai kết quả test riêng, không gộp thành một F1 chung. Mỗi hành vi chỉ có 1 positive event và 1 negative clip trong từng test split.

P01: loitering và suspicious behavior đạt F1 = 1.000 với 0 false-alarm clip âm. P02: F1 tương ứng là 0.500 và 0.667; intrusion, theft, line crossing đều F1 = 0.000. P01 là pilot v2 còn P02 là pilot v3, nên không thể quy chênh lệch chỉ cho khác người tham gia hoặc suy rộng F1 = 1.000 của P01 ra toàn bộ 40 clip. `tp` là alert khớp sự kiện, `fp` đếm alert không khớp ground truth (kể cả alert lặp trên clip dương), và `fn` là sự kiện bị bỏ sót; `false_alarm_rate_negative_clip` chỉ xét clip âm.

Nguồn nội bộ: `experiments/results/runtime_2026-08-17/summary.csv`, `experiments/priority_19_08/behavior_evaluation/frozen_test/evaluation/behavior_metrics.csv`, và `experiments/reviewer_expansion_20260908/p02_evaluation_test/behavior_metrics.csv`.
