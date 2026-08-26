# Two-identity face recognition pilot

## Phạm vi và giao thức

- Dataset: 24 ảnh của 2 danh tính (Nguyen: 11; Tuan: 13).
- Enrollment: 6 ảnh (3 ảnh/danh tính).
- Development: 10 ảnh; không dùng để tính frozen metric.
- Frozen test: 8 ảnh (Nguyen: 5; Tuan: 3).
- Stack production: InsightFace `buffalo_sc`, detection size `320`, cùng cơ chế detect/align của `FaceAnalysis`.
- Embedding được L2-normalize; cosine similarity được tính bằng dot product.
- Mỗi ảnh test được so với 3 enrollment references của mỗi identity; score identity là giá trị lớn nhất, đúng aggregation `max` của runtime.
- Số score dùng cho metric: 8 genuine và 8 impostor (tương ứng 48 phép so sánh ảnh-tham chiếu trước aggregation).
- Tất cả 14 ảnh enrollment/frozen-test đều detect được mặt và trích được embedding; chi tiết ở `face_embedding_audit.csv`.

## Kết quả tại các ngưỡng đang quan tâm

| Threshold | FAR | FRR | TPR | FPR | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.000 | 0.250 | 0.750 | 0.000 | 6 | 0 | 8 | 2 |
| 0.40 | 0.000 | 0.500 | 0.500 | 0.000 | 4 | 0 | 8 | 4 |
| 0.45 | 0.000 | 0.750 | 0.250 | 0.000 | 2 | 0 | 8 | 6 |

## EER và ngưỡng pilot

- EER gần đúng: **0.125** tại threshold **0.270** (nearest point on 0.005 threshold grid; FAR=0.125, FRR=0.125).
- Best-performing threshold trên lưới sweep của pilot: **0.335**, balanced accuracy=0.938, FAR=0.000, FRR=0.125.
- Trong ba operating point 0.35/0.40/0.45, **0.35** cho kết quả tốt nhất (FAR=0.000, FRR=0.250); 0.40 và 0.45 làm FRR tăng lần lượt lên 0.500 và 0.750.
- Kết luận chỉ áp dụng cho two-identity pilot này; không khẳng định threshold tối ưu toàn cục.

## Hạn chế

- Chỉ có 2 identities và sample size nhỏ.
- Không participant-disjoint ngoài hai danh tính này và không đại diện cho dân số lớn.
- Ảnh enrollment/test khác filename và appearance/session theo mô tả, nhưng quy mô chưa đủ cho full-scale validation.
- Kết quả chỉ kiểm tra tính hợp lý của threshold hiện tại; cần thêm nhiều danh tính, session, điều kiện sáng, góc mặt và thiết bị trước khi suy rộng.

## Artifact integrity

- Split được khóa trước khi tính metric trong `FACE_PILOT_FREEZE.yaml` và `face_pilot_split.csv`.
- Không augment, không thay model, không retune ảnh và không thay frozen-test split sau khi xem kết quả.
