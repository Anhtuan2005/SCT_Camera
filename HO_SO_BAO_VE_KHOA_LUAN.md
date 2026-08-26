# HỒ SƠ CHUẨN BỊ BẢO VỆ KHÓA LUẬN - SCT CAMERA

> Đề tài: **Xây dựng hệ thống cảnh báo thời gian thực cho camera giám sát an ninh dựa trên AI**  
> Mục tiêu của tài liệu: giúp trình bày đúng sản phẩm, nhớ đúng số liệu, trả lời phản biện có bằng chứng và không tuyên bố vượt quá phạm vi thí nghiệm.

## 1. Bản tóm tắt phải thuộc lòng

### 1.1. Một câu mô tả đề tài

SCT Camera biến webcam, video hoặc RTSP thành một hệ thống giám sát thời gian thực có phát hiện và theo dõi đối tượng, phân giải danh tính, phân tích hành vi, cảnh báo ngã, gửi cảnh báo đa kênh và quản lý qua dashboard web trên một máy tính có GPU phổ thông.

### 1.2. Bài nói mở đầu khoảng 45 giây

> Camera truyền thống chủ yếu ghi hình hoặc báo chuyển động nên người vận hành vẫn phải tự xem và thường gặp nhiều cảnh báo ít giá trị. Nhóm xây dựng SCT Camera theo pipeline sáu tầng: thu nhận video; phát hiện và theo dõi; danh tính và tư thế; phân tích hành vi; ghi nhận/chấm điểm rủi ro; giao diện và cảnh báo. Sản phẩm tích hợp YOLO11s, ByteTrack có bộ nhớ track và bù chuyển động camera, InsightFace, các luật hình học, FastAPI và cơ chế cảnh báo bất đồng bộ. Hệ thống đã được đo trên hai GPU với một đến bốn camera, chạy hai phiên bốn camera trong tám giờ và đạt 206/206 kiểm thử tự động. Tuy nhiên, nhóm chỉ xem kết quả hành vi và khuôn mặt là pilot nhỏ; lớp Risk chưa có đủ nhãn và model hợp lệ nên không tuyên bố đã giảm cảnh báo giả.

### 1.3. Đóng góp nên trình bày

1. Xây dựng được sản phẩm đầu-cuối, không chỉ một notebook hoặc một mô hình đơn lẻ.
2. Tách capture/display khỏi nhịp phân tích AI để giao diện không bị khóa theo thời gian suy luận.
3. Mở rộng ByteTrack bằng bộ nhớ track mất tạm thời, bù chuyển động camera và lọc box trùng.
4. Kết hợp danh tính, tư thế và các luật hành vi có thể cấu hình theo camera.
5. Xây dựng hạ tầng ghi 22 đặc trưng, gán nhãn và đánh giá Risk; trung thực giữ Risk gate ở trạng thái tắt khi chưa đủ dữ liệu.
6. Công bố cả kết quả tốt lẫn lỗi trên frozen test, kèm hash video/cấu hình và không retune sau khi xem test.

### 1.4. Điều không được tuyên bố

- Không nói “độ chính xác hệ thống là 100%” chỉ vì 206/206 testcase đạt.
- Không nói Risk đã giảm cảnh báo giả; chưa có nhãn hợp lệ, chưa có model `.npz`, `gate_alerts=false`.
- Không nói hệ thống đã được kiểm chứng tổng quát; pilot hành vi có một người và mỗi hành vi chỉ có một clip dương, một clip âm trong test.
- Không nói ngưỡng khuôn mặt 0,35 là tối ưu toàn cục; nó chỉ tốt nhất trong pilot hai danh tính đang có.
- Không nói đã có mAP, MOTA, IDF1 hoặc HOTA của sản phẩm; dự án chưa có ground truth box/trajectory phù hợp.
- Không nói production intrusion hiện tại là cảnh báo polygon; phiên bản đóng băng chỉ phát `intrusion` khi cắt vạch theo hướng `IN`.
- Không nói YOLO11 là “mới nhất” hoặc “tốt nhất”; hãy nói đây là lựa chọn phù hợp với stack và phép đo của đề tài.

## 2. Khung thuyết trình 10-12 phút

| Thời gian | Nội dung | Thông điệp phải chốt |
|---:|---|---|
| 0:00-0:45 | Vấn đề và động lực | Camera ghi hình thụ động chưa hiểu đối tượng, danh tính và hành vi. |
| 0:45-1:30 | Mục tiêu và phạm vi | Hộ gia đình/cửa hàng nhỏ, một đến vài camera, một máy có GPU phổ thông. |
| 1:30-2:30 | Kiến trúc | Capture → Detection/Tracking → Identity/Pose → Analytics → Learning/Risk → Web/Alert. |
| 2:30-3:30 | Phát hiện và tracking | YOLO11s + ByteTrack; bộ nhớ track, CMC và lọc trùng là phần mở rộng của nhóm. |
| 3:30-4:30 | Danh tính và hành vi | InsightFace; trạng thái pending/known/stranger; luật ROI/vạch; fall safety-critical. |
| 4:30-5:15 | Cảnh báo và giao diện | Producer-consumer, queue bất đồng bộ, cooldown/dedup, Telegram/Discord/siren, FastAPI. |
| 5:15-6:30 | Hiệu năng/scaling | Nêu hai GPU, 1-4 camera, giải thích aggregate AI FPS và scaling efficiency. |
| 6:30-7:30 | Kiểm thử và stability | 206/206 pytest; hai phiên 4 camera × 8 giờ; giới hạn của bằng chứng. |
| 7:30-9:00 | Pilot hành vi và khuôn mặt | Nêu cả TP, FP, FN; không giấu theft/duplicate/false crossing. |
| 9:00-10:00 | Hạn chế | Risk chưa xác thực, dữ liệu nhỏ, thiếu MOT/mAP benchmark, chưa suy rộng RTSP/>4 camera. |
| 10:00-11:00 | Hướng phát triển | Sửa dedup/theft/debounce; participant-disjoint; ablation; MOT benchmark; batching. |
| 11:00-12:00 | Demo/kết luận | Chứng minh luồng sản phẩm; nhắc lại tính khả thi thay vì tuyên bố hoàn thiện tuyệt đối. |

### Câu kết luận nên dùng

> Kết quả của đề tài chứng minh tính khả thi của một pipeline giám sát AI tích hợp trên phần cứng phổ thông trong phạm vi đã đo. Đóng góp chính là sản phẩm đầu-cuối, kiến trúc vận hành, các cải tiến tracking và quy trình đánh giá trung thực. Những phần chưa đủ dữ liệu, đặc biệt Risk và khả năng khái quát, được nhóm giữ ở trạng thái chưa kết luận và đã xác định giao thức đánh giá tiếp theo.

## 3. Kiến trúc sản phẩm cần hiểu thật chắc

### 3.1. Luồng dữ liệu

```text
Webcam / video / RTSP
  → OpenCV capture
  → một CameraPipeline cho mỗi camera
  → YOLO11s detection
  → ByteTrack + track memory + CMC + duplicate suppression
  → YOLO11n-pose khi cần
  → InsightFace identity
  → BehaviorEngine + FallDetector
  → BehaviorLearningService ghi 22 đặc trưng / chấm Risk nếu có model
  → vẽ annotation + FrameBuffer
  → MJPEG dashboard
  → AlertManager queue
  → cooldown/dedup
  → Telegram / Discord / siren / SQLite
```

### 3.2. Vì sao tách capture/display khỏi AI analysis

- Mỗi camera chỉ có một analysis job đang chạy, tránh tạo vô hạn worker khi AI chậm.
- `frame_skip` và `ai_max_fps` khống chế tần suất phân tích.
- Khi AI đang chạy, pipeline vẫn phát frame hiển thị với snapshot phân tích gần nhất.
- `analysis_stale_after_ms` giúp phát hiện kết quả AI đã cũ.
- `analysis_timeout_min_seconds` ngăn sinh thêm worker trùng khi analysis có dấu hiệu treo.
- Gửi thông báo mạng chạy bất đồng bộ nên độ trễ Telegram/Discord không chặn camera.

### 3.3. Quyền sở hữu state

- `RuntimeState` quản lý cấu hình, detector dùng chung, pose estimator, behavior engine, identity resolver, alert manager, frame buffer và các pipeline.
- Mỗi camera có tracker riêng; detector và khóa suy luận được dùng chung.
- Cấu hình camera, ROI và line được quản lý qua API/UI; không nên để route tự chạm vào nội bộ pipeline.

### 3.4. Cơ chế chịu tải

- Nguồn video cục bộ có thể đọc nhanh hơn thời gian thực, vì vậy báo cáo dùng `aggregate_ai_fps`, không dùng tốc độ vòng đọc video.
- Một nguồn camera không được phép làm nghẽn luồng cảnh báo hoặc camera khác.
- Queue cảnh báo có dung lượng 1.000; cảnh báo mức cao khi đạt 800.
- Cooldown/dedup dùng khóa gồm camera, loại cảnh báo, vùng/vạch/track.

## 4. Kiến thức lý thuyết phải chuẩn bị

### 4.1. YOLO và object detection

Phải giải thích được:

- YOLO là detector một giai đoạn: dự đoán lớp và box trong một lượt suy luận, phù hợp thời gian thực.
- `confidence` là độ tin cậy của dự đoán; hạ ngưỡng tăng recall nhưng có thể tăng false positive.
- IoU đo độ chồng lấp: `IoU = diện tích giao / diện tích hợp`.
- NMS loại các box trùng có cùng đối tượng dựa trên confidence và IoU.
- mAP cần ground truth box/class; dự án chưa có annotation phù hợp nên không được tự suy ra mAP từ cảnh báo.
- Bản final dùng `yolo11s.pt` cho detection và `yolo11n-pose.pt` cho pose.

**Vì sao chọn YOLO11s:** cân bằng chất lượng-tốc độ tốt hơn bản nano trong bối cảnh góc rộng/vật thể nhỏ, có sẵn trong Ultralytics và cùng hệ sinh thái với pose/tracking/export. Đổi lại, nó nặng hơn YOLO11n và thông lượng giảm khi số camera tăng.

### 4.2. ByteTrack

ByteTrack là tracking-by-detection. Điểm cốt lõi là không bỏ hoàn toàn detection score thấp: lượt đầu ghép detection tin cậy cao, lượt sau dùng detection thấp để phục hồi đối tượng thật bị che khuất và lọc nền.

**Vì sao chọn thay DeepSORT:**

- Không cần một mạng ReID ngoại hình riêng trong vòng tracking nên nhẹ hơn.
- Phù hợp phần cứng phổ thông và pipeline nhiều camera.
- Khôi phục tốt hơn các track có confidence giảm tạm thời.
- Nhược điểm: khi che khuất dài hoặc nhiều người giống nhau, thiếu appearance embedding có thể làm ID switch; vì vậy nhóm bổ sung track memory nhưng vẫn chưa thay thế đánh giá MOT chuẩn.

### 4.3. Kalman và Hungarian

- Kalman dự đoán trạng thái/vị trí tiếp theo từ lịch sử chuyển động và cập nhật khi có quan sát mới.
- Hungarian giải bài toán gán toàn cục giữa track dự đoán và detection theo ma trận chi phí, tốt hơn greedy khi các lựa chọn cạnh tranh nhau.
- Khi camera rung, dự đoán chuyển động đối tượng bị lẫn với chuyển động toàn cảnh; CMC ước lượng chuyển động toàn cục để bù trước khi ghép.

### 4.4. CMC bằng sparse optical flow

1. Chọn các điểm đặc trưng ở hai frame liên tiếp.
2. Theo dõi dịch chuyển của các điểm bằng optical flow.
3. Ước lượng phép biến đổi toàn cục của camera.
4. Bù phép biến đổi này khi dự đoán/ghép track.
5. Nếu ước lượng thất bại, fallback về identity transform để pipeline vẫn chạy.

CMC không bảo đảm hết ID switch; nó chủ yếu xử lý rung/pan nhẹ và phụ thuộc số lượng/chất lượng điểm đặc trưng.

### 4.5. InsightFace, ArcFace và cosine similarity

- InsightFace detect/align khuôn mặt rồi tạo embedding.
- ArcFace dùng additive angular margin để tạo embedding cùng người gần nhau, khác người tách nhau trên hypersphere.
- Embedding được chuẩn hóa L2; cosine similarity sau đó có thể tính bằng tích vô hướng.
- Runtime có ba trạng thái chính: `pending_person`, `known_person`, `stranger`.
- Xác nhận nhiều lần và memory theo không gian/thời gian giúp giảm nhấp nháy nhãn khi track ID thay đổi.

**Đánh đổi ngưỡng:** ngưỡng cao giảm nhận nhầm nhưng tăng từ chối người thật; ngưỡng thấp tăng nhận đúng nhưng tăng nguy cơ impostor được chấp nhận. Cấu hình global hiện tại là 0,45; pilot nhỏ cho thấy 0,35 tốt nhất trong ba điểm 0,35/0,40/0,45. Không đổi kết luận thành “0,35 tối ưu toàn cục”.

### 4.6. Hình học hành vi

- Point-in-polygon: kiểm tra tâm box có nằm trong ROI.
- Cắt vạch: dùng dấu của tích có hướng ở hai thời điểm; đổi phía hoặc đoạn chuyển động giao với vạch cho biết crossing.
- Khoảng cách người-tài sản được chuẩn hóa theo đường chéo frame để giảm phụ thuộc độ phân giải.
- Quãng đường, dịch chuyển thuần và số lần pacing được tính từ `center_history`.

**Semantics phải nói đúng:** trong code đóng băng, `intrusion` là cảnh báo khi cắt line theo hướng `IN`. Counter ghi cả `IN/OUT`; production không phát loại cảnh báo `line_crossing` riêng. Runner thí nghiệm chỉ quan sát delta counter để đánh giá line crossing độc lập.

### 4.7. Fall detection

- Không kết luận từ một frame; cần chuyển từ đứng/đi/chạy sang lying hoặc changing-to-lying.
- Sau khoảng 10 giây vẫn nằm: `possible_fall`.
- Khoảng 45 giây: nâng thành `possible_unresponsive`/POSSIBLE EMERGENCY.
- Khi che khuất sau khi đã quan sát lying, giữ candidate và `LAST SEEN` tối đa 60 giây.
- Hồi phục cần tư thế ngồi/đứng/đi/chạy ổn định 3 giây.
- Cảnh báo ngã không bị Risk gate chặn.
- Đây là cảnh báo sớm, không phải chẩn đoán bất tỉnh, đột quỵ hoặc kết luận y tế.

### 4.8. Logistic Regression và 22 đặc trưng Risk

Hàm chấm điểm dự kiến:

```text
z_i = (x_i - mean_i) / std_i
logit = w·z + b
risk = sigmoid(logit)
```

Nhóm 22 đặc trưng:

1. Thời gian: `duration`, `threshold_seconds`, `duration_ratio`, `near_seconds`.
2. Điểm luật: `score`, `score_ratio`, `pacing_passes`.
3. Detection/hình học: `object_confidence`, `bbox_area_ratio`, `path_length_ratio`, `net_distance_ratio`, `displacement_ratio`, `speed_ratio`.
4. Tín hiệu theft: `has_vehicle_signal`, `vehicle_started_moving`, `moving_same_direction`, `pose_push_contact`.
5. Ngữ cảnh/lớp: `object_count`, `zone_configured`, `is_person`, `is_vehicle`, `is_asset`.

**Vì sao Logistic Regression:** phù hợp dữ liệu dạng bảng, nhẹ, dễ giải thích trọng số, suy luận rất rẻ. Nhưng hiện chưa có nhãn dương/âm hợp lệ và chưa có model, vì vậy mới chỉ có hạ tầng chứ chưa có hiệu quả định lượng.

### 4.9. Các metric

```text
Precision = TP / (TP + FP)
Recall hoặc Detection Rate = TP / (TP + FN)
F1 = 2 × Precision × Recall / (Precision + Recall)
FAR trong pilot = số clip âm có ít nhất một false alert / tổng clip âm
Scaling efficiency(N) = FPS_N / (N × FPS_1)
```

- Precision trả lời: trong các cảnh báo phát ra, bao nhiêu cảnh báo đúng.
- Recall trả lời: trong các sự kiện thật, phát hiện được bao nhiêu.
- FAR của pilot dùng đơn vị clip âm, trong khi FP/precision dùng đơn vị event; không được trộn hai đơn vị.
- Latency tính từ lúc hành vi đủ điều kiện theo ground truth đến TP đầu tiên.
- MOTA, IDF1 và HOTA cần ground-truth trajectory. HOTA cân bằng detection, association và localization; đây là hướng đánh giá tương lai, chưa phải kết quả hiện có.

## 5. Bảng số liệu phải nhớ

### 5.1. Môi trường

| Thành phần | Laptop | GPU VM |
|---|---|---|
| OS | Windows 11 64-bit | Windows 10 Pro build 19045 |
| CPU | Ryzen 7 6800H | Core i7-12700K theo cấu hình thuê |
| RAM | 16 GB | 28 GB |
| GPU | RTX 3050 Laptop, 4 GB | RTX 5060 Ti, 16 GB |
| Model | YOLO11s + YOLO11n-pose + buffalo_sc | Cùng stack |
| Dữ liệu scaling | 4 video cục bộ | Cùng 4 video |

### 5.2. Scaling 10 phút

| GPU | 1 cam | 2 cam | 3 cam | 4 cam | Hiệu suất scaling 4 cam |
|---|---:|---:|---:|---:|---:|
| RTX 3050 Laptop | 7,56 | 16,35 | 21,34 | 16,40 | 54,2% |
| RTX 5060 Ti | 7,60 | 15,31 | 21,97 | 29,83 | 98,2% |

Diễn giải:

- Đây là **aggregate AI FPS**, không phải FPS cho từng camera.
- Nếu chia đều ở bốn camera: RTX 3050 khoảng 4,10 AI FPS/camera; RTX 5060 Ti khoảng 7,46 AI FPS/camera.
- RTX 3050 đạt đỉnh ở ba camera rồi giảm ở bốn camera, gợi ý nghẽn CPU/lock/scheduling chứ không chỉ VRAM.
- Không suy rộng kết quả sang GPU khác, hơn bốn camera hoặc RTSP.

### 5.3. Độ trễ một camera

| Giai đoạn | p50 | p95 | p99 |
|---|---:|---:|---:|
| Detection + Tracking | 19,97 ms | 27,21 ms | 34,81 ms |
| Pose | 18,79 ms | 25,91 ms | 32,17 ms |
| Identity state | 0,12 ms | 0,17 ms | 0,22 ms |
| Behavior | 1,15 ms | 1,70 ms | 2,45 ms |

Lưu ý: identity latency ở đây không bao gồm đầy đủ live InsightFace vì video benchmark dùng chính sách gán người lạ. Không lấy 0,12 ms làm thời gian nhận diện khuôn mặt đầy đủ.

### 5.4. Test và stability

- Bộ hiện tại đã được chạy lại: **206 passed, 3 Starlette deprecation warnings**.
- Nhóm fall/pose/pipeline/drawing mục tiêu: 72/72, là tập con của 206, không cộng thành 278.
- Hai phiên stability độc lập: bốn camera × 8 giờ × 2.880 mẫu.
- RTX 3050: AI FPS TB 24,05; 4/4 online; drop 0; Tmax 57 °C.
- RTX 5060 Ti: AI FPS TB 29,49; 4/4 online; drop 0; Tmax 52 °C.
- Logger RTX 5060 Ti có khoảng trống 35,5 giây; trạng thái hai phía vẫn 4/4 online và không giảm thông lượng bất thường.
- Kết luận đúng: “không quan sát thấy memory leak, throughput drift hoặc thermal throttling trong hai phiên đã thử”, không phải “hệ thống không bao giờ rò rỉ”.

### 5.5. Pilot hành vi frozen test

| Hành vi | TP/FP/TN/FN | DR/FAR | Precision/Recall/F1 | Latency |
|---|---|---|---|---:|
| Intrusion | 1/5/1/0 | 1,000/0,000 | 0,166667/1,000/0,285714 | 0,457 s |
| Loitering | 1/0/1/0 | 1,000/0,000 | 1,000/1,000/1,000 | 0,725 s |
| Suspicious behavior | 1/0/1/0 | 1,000/0,000 | 1,000/1,000/1,000 | 0,186 s |
| Theft | 0/1/0/1 | 0,000/1,000 | 0,000/0,000/0,000 | N/A |
| Line crossing | 1/2/0/0 | 1,000/1,000 | 0,333333/1,000/0,500000 | 4,084 s |

Phạm vi: 20 clip của P01, gồm 10 development và 10 frozen test; tổng 1.344,240 giây. Mỗi hành vi trong test chỉ có một clip dương và một clip âm. Split theo repetition/session nhưng không participant-disjoint.

### 5.6. Pilot khuôn mặt

- 24 ảnh, 2 identities; 6 enrollment; 10 development; 8 frozen test.
- 8 genuine và 8 impostor score sau aggregation; 48 phép so sánh ảnh-tham chiếu trước aggregation.
- EER gần đúng 0,125 tại threshold 0,270.
- Threshold 0,35: FAR 0,000; FRR 0,250; TP=6, FP=0, TN=8, FN=2.
- Threshold 0,40: FAR 0,000; FRR 0,500.
- Threshold 0,45: FAR 0,000; FRR 0,750.
- Best grid point của pilot là 0,335, balanced accuracy 0,9375; không coi là ngưỡng production đã được xác minh rộng.

## 6. Các mâu thuẫn phiên bản phải chủ động xử lý

### 6.1. Intrusion polygon hay line IN

**Hiện trạng:** báo cáo có đoạn 3.5 mô tả intrusion khi vào vùng cấm, nhưng code, paper và frozen protocol xác nhận `IntrusionDetector.analyze()` chỉ gọi `_analyze_lines()`. `_analyze_zones()` tồn tại nhưng không được gọi.

**Câu trả lời:**

> Ở phiên bản code đóng băng dùng cho thí nghiệm, alert production `intrusion` là tập con hướng IN của cơ chế cắt vạch. Hàm polygon còn tồn tại như code chưa được nối vào luồng gọi. Đoạn mô tả theo vùng trong báo cáo là semantics cũ/chưa đồng bộ; khi trình bày kết quả em lấy code và frozen artifact làm nguồn sự thật và không báo metric polygon intrusion.

### 6.2. Lịch sử cảnh báo trong RAM hay SQLite

**Hiện trạng:** phần hạn chế của báo cáo nói lịch sử chỉ ở RAM, nhưng code hiện tại truyền `DatabaseManager` vào `AlertManager`; lịch sử được `persist_alert` và API đọc từ SQLite, RAM deque chỉ là fallback/cache gần.

**Câu trả lời:**

> Đoạn hạn chế đó đã cũ so với code hiện tại. Phiên bản đang demo chạy migration khi startup, ghi alert và trạng thái delivery vào SQLite, và API ưu tiên đọc lịch sử từ SQLite. Phần còn thiếu thực sự là video evidence trước/sau sự kiện, không phải toàn bộ alert history.

### 6.3. YOLO11n trong paper và YOLO11s trong báo cáo final

**Hiện trạng:** paper giữ một phép đo cũ với YOLO11n và 1-2 camera; báo cáo final và cấu hình hiện tại dùng YOLO11s, phép đo 1-4 camera trên hai GPU.

**Câu trả lời:**

> Đây là hai artifact thí nghiệm khác nhau, khác model/cấu hình/video nên không được so trực tiếp. Khi bảo vệ sản phẩm final em dùng bảng YOLO11s 1-4 camera trong báo cáo; phép đo YOLO11n trong paper chỉ dùng để mô tả phiên bản paper đã đóng băng.

### 6.4. Bảy E2E đều PASSED nhưng pilot có lỗi

**Không mâu thuẫn:** E2E chứng minh một kịch bản cụ thể có thể chạy đúng. Frozen pilot tính cả clip âm, event trùng, miss và latency nên nghiêm ngặt hơn.

> Một ca E2E passed không phải ước lượng xác suất đúng. Pilot định lượng mới cho thấy theft chưa ổn và intrusion/line crossing còn lỗi. Vì vậy nhóm giữ cả hai kết quả thay vì dùng ảnh demo để thay thế metric.

### 6.5. 206 testcase và “độ chính xác”

206 testcase kiểm tra logic phần mềm, rollback, threshold, state, dedup, pipeline và web. Nó không thay thế ground truth ngoài đời, mAP, MOT hoặc behavior accuracy.

### 6.6. Risk đã “xây dựng” nhưng chưa “xác thực”

Đã có feature extraction, event log, label API/CSV, trainer/evaluator, load/hot-reload và gate. Chưa có positive/negative label hợp lệ và model nên chỉ được nói “hạ tầng Risk đã hoàn thiện về luồng kỹ thuật”, không nói “Risk hoạt động hiệu quả”.

### 6.7. Trạng thái mã nguồn

- Git HEAD hiện là `a7881f78`, nhưng kết quả 206 test gắn với working tree có thay đổi cục bộ.
- Frozen pilot có hash file, video và cấu hình nên vẫn có bằng chứng tái lập thí nghiệm.
- Trước ngày bảo vệ nên tạo một bản sao read-only hoặc commit/tag đúng trạng thái demo, kèm `requirements`, model hash và config đã che bí mật.

### 6.8. Hai số bốn-camera trên RTX 3050 không khớp nhau

**Hiện trạng:** phép scaling 10 phút báo 16,40 aggregate AI FPS ở bốn camera, trong khi phiên stability 8 giờ báo trung bình 24,05. Báo cáo chưa lưu ngay cạnh hai bảng đủ metadata để chứng minh hai lần chạy thật sự dùng cùng model, `frame_skip`, `ai_max_fps`, source và revision.

**Cách xử lý:** không gộp hai số thành một đường benchmark. Trước buổi bảo vệ phải tìm raw CSV/config của phiên stability và đối chiếu model, code hash, source, scheduler settings. Nếu không khôi phục được đầy đủ, hãy trình bày đây là hai lần chạy khác cửa sổ/cấu hình và chỉ dùng phiên 8 giờ để nói về xu hướng theo thời gian, không dùng nó để phủ định bảng scaling 10 phút.

### 6.9. Cách đếm “sáu tầng” chưa thống nhất giữa các sơ đồ

Một số hình tách Detection và Tracking thành hai box nhưng không vẽ Learning/Risk; paper lại gộp hai thành Vision AI và vẽ Learning/Risk riêng. Trên slide bảo vệ nên dùng đúng một taxonomy: **Capture; Vision AI (Detection + Tracking + Pose); Identity; Analytics; Learning/Risk; Web/Alert**. Nói rõ đây là sáu tầng khái niệm; các box trong code có thể tách nhỏ hơn và Alert/Web nằm sau ranh giới thread-safe/async.

## 7. Kịch bản demo an toàn

### 7.1. Demo 5-6 phút

1. Mở dashboard và đăng nhập.
2. Chỉ ra trạng thái camera, FPS/latency và luồng MJPEG.
3. Mở một camera dùng video local đã kiểm tra trước.
4. Cho thấy ROI/line editor và giải thích normalized coordinates.
5. Chạy một clip dương ngắn: line IN, loitering hoặc fall; không chọn theft làm demo chính.
6. Cho thấy annotation, timer/counter và bản ghi alert.
7. Mở alert history từ API/UI để chứng minh SQLite persistence.
8. Nếu mạng ổn, gửi Telegram/Discord test; nếu không, dùng ảnh minh chứng có sẵn.
9. Kết thúc bằng bảng số liệu frozen test, không lấy riêng clip demo làm bằng chứng định lượng.

### 7.2. Checklist trước khi vào hội đồng

- Đổi password mặc định; không chiếu token, RTSP credential hoặc ảnh khuôn mặt chưa được phép.
- Giữ sẵn toàn bộ weight local, tránh tải model trong phòng bảo vệ.
- Pre-warm detector/pose/InsightFace trước giờ trình bày.
- Dùng video local; có camera thật nhưng không phụ thuộc hoàn toàn vào camera thật.
- Chuẩn bị một video quay màn hình demo, ảnh Telegram và screenshot dashboard làm phương án B.
- Tắt ứng dụng nền nặng, cắm nguồn, chọn chế độ hiệu năng cao và kiểm tra CUDA.
- Chạy `python -m pytest -q` trước buổi bảo vệ; lưu screenshot “206 passed”.
- Ghi rõ demo config dùng threshold nào; không âm thầm đổi threshold rồi dùng metric của cấu hình khác.
- Có bản CSV/ảnh của scaling, behavior pilot, face pilot và stability để mở khi bị hỏi.

### 7.3. Nếu demo lỗi

- Camera không mở: chuyển sang video local đã cấu hình.
- Không có CUDA: nói rõ hệ thống fallback CPU nhưng demo thời gian thực sẽ giảm; dùng video quay sẵn.
- Telegram/Discord mất mạng: cho thấy queue, SQLite history và ảnh lần chạy trước; không mất thời gian sửa mạng trên sân khấu.
- Model chưa warm: giải thích startup/warm-up nhưng chuyển ngay sang clip quay sẵn.
- Alert không xuất hiện: kiểm tra identity state, ROI/line, threshold và timer; không đổi ngưỡng tùy tiện trước hội đồng.

## 8. Công thức trả lời phản biện

Mỗi câu khó trả lời theo bốn lớp:

1. **Trả lời trực tiếp:** có/không/chọn gì.
2. **Bằng chứng:** code, bảng, metric hoặc artifact nào.
3. **Phạm vi:** kết luận áp dụng đến đâu.
4. **Bước tiếp theo:** cần dữ liệu/thí nghiệm/sửa lỗi gì.

Ví dụ:

> Hệ thống chưa chứng minh Risk giảm false alarm. Bằng chứng là label hiện có đều trống, model chưa tồn tại và gate đang tắt. Vì vậy nhóm chỉ công bố hạ tầng 22 feature và evaluator, không công bố Accuracy/F1/ROC-AUC. Bước tiếp theo là thu thập nhãn có video evidence, chia group-disjoint, chọn threshold trên validation và chỉ báo metric trên test.

## 9. Ngân hàng câu hỏi phản biện và câu trả lời mẫu

### A. Bài toán và đóng góp

**1. Đề tài mới ở đâu khi các thư viện đều có sẵn?**  
Điểm mới không phải phát minh YOLO hay ByteTrack. Đóng góp là thiết kế và hiện thực hệ thống đầu-cuối có state theo thời gian, cấu hình nhiều camera, danh tính, hành vi, cảnh báo, UI, đo lường và quy trình frozen evaluation. Phần kỹ thuật riêng gồm track memory, CMC, duplicate suppression và tích hợp fall/behavior/risk logging.

**2. Đây là nghiên cứu hay chỉ là tích hợp sản phẩm?**  
Là khóa luận theo hướng AI ứng dụng/system engineering. Có giả thuyết và đánh giá cho scaling, latency, stability, behavior và face threshold; tuy nhiên chưa có benchmark chuẩn để tuyên bố cải tiến thuật toán tracking. Nhóm tách rõ phần đã chứng minh và phần cần ablation/MOT benchmark.

**3. Ai là người dùng chính?**  
Người vận hành camera tại hộ gia đình, cửa hàng nhỏ hoặc cơ sở quy mô nhỏ, cần xem trạng thái, cấu hình ROI/line và nhận cảnh báo mà không phải theo dõi màn hình liên tục.

**4. Phạm vi không bao gồm gì?**  
Không phải hệ thống thành phố/hàng trăm camera; chưa xác thực mọi GPU, hơn bốn camera, RTSP diện rộng, mọi dân số khuôn mặt hoặc tình huống y tế.

**5. Giá trị lớn nhất của đề tài là gì?**  
Một sản phẩm có thể chạy và đo được, đồng thời có kiến trúc đủ tách lớp để kiểm thử/thay thế module và có bằng chứng về giới hạn thay vì chỉ trình diễn ca thành công.

### B. Lựa chọn công nghệ

**6. Vì sao YOLO11s thay vì YOLO11n?**  
YOLO11s cân bằng recall/chất lượng tốt hơn cho camera góc rộng và vật thể nhỏ; vẫn chạy được trên RTX 3050 4 GB. Đổi lại, nó nặng hơn, nên scaling laptop giảm ở bốn camera.

**7. Vì sao không dùng RT-DETR/DEIM?**  
YOLO11 có stack ổn định, detection/pose/tracking và export sẵn, phù hợp thời gian và phần cứng đề tài. RT-DETR/DEIM là baseline/hướng phát triển hợp lý, nhưng thay detector phải được benchmark cùng dữ liệu và không nên thực hiện chỉ vì mới hơn.

**8. Vì sao ByteTrack thay DeepSORT?**  
ByteTrack nhẹ hơn vì không yêu cầu ReID network cho mọi detection, tận dụng box confidence thấp để giữ track qua che khuất. Nhược điểm appearance yếu hơn đã được thừa nhận; track memory chỉ hỗ trợ che khuất ngắn, không thay thế DeepSORT trong cảnh đông người.

**9. Track memory có thể ghép nhầm không?**  
Có. Cửa sổ quá dài hoặc ngưỡng vị trí/kích thước quá rộng sẽ nối người mới vào ID cũ. Vì vậy cần ablation và ground-truth trajectory để chọn tham số theo IDF1/ID switches, không chỉ xem bằng mắt.

**10. CMC giúp gì và khi nào không giúp?**  
CMC loại chuyển động toàn cảnh do rung/pan nhẹ khỏi chuyển động đối tượng. Nó kém khi cảnh ít texture, motion blur nặng, parallax lớn hoặc chuyển động camera phi tuyến; code fallback identity transform khi ước lượng lỗi.

**11. Vì sao dùng FastAPI + MJPEG thay WebRTC?**  
MJPEG đơn giản, dễ tích hợp với browser và đủ cho dashboard nội bộ. Nhược điểm là bandwidth và latency kém WebRTC, không có adaptive streaming; WebRTC phù hợp hơn nếu triển khai mạng lớn.

**12. Vì sao SQLite?**  
Một node đơn, cài đặt đơn giản, transaction/migration rõ ràng và đủ cho lịch sử cảnh báo quy mô đề tài. Nếu đa node hoặc lưu lượng lớn, cần PostgreSQL/message broker/object storage.

### C. Luồng thời gian thực

**13. Một camera AI chậm có chặn camera khác không?**  
Mỗi camera có capture thread và analysis slot riêng; network alert chạy ở async worker. Tuy nhiên detector/pose dùng khóa suy luận chung nên tải AI vẫn cạnh tranh và làm tăng latency, thể hiện ở scaling.

**14. Vì sao không phân tích mọi frame?**  
Không cần cho các hành vi có ngưỡng giây; phân tích mọi frame gây lãng phí. `frame_skip` và `ai_max_fps` giữ tải ổn định, trong khi display tiếp tục dùng frame mới và snapshot AI gần nhất.

**15. Kết quả AI cũ có gây sai không?**  
Có thể. Vì vậy pipeline đo tuổi snapshot và cảnh báo stale. Thiết kế này ưu tiên UI liên tục; với ứng dụng safety-critical cần deadline rõ, drop policy và đo end-to-end event latency chặt hơn.

**16. Tại sao tổng p50 khoảng 40 ms nhưng AI FPS một camera chỉ 7,56?**  
Stage latency là thời gian service của một analysis job; AI FPS còn bị giới hạn bởi `ai_max_fps`, `frame_skip`, capture scheduling, lock dùng chung và pipeline overhead. Không thể lấy nghịch đảo đơn giản của tổng p50 để suy ra throughput cấu hình.

**17. Aggregate AI FPS là gì?**  
Tổng số lượt phân tích AI mỗi giây trên tất cả camera. Nó không phải display FPS và không phải FPS của từng camera.

### D. Danh tính và quyền riêng tư

**18. Cosine threshold 0,45 có căn cứ gì?**  
Đó là cấu hình hiện tại, không phải optimum được chứng minh. Pilot so ba mức cho thấy 0,35 tốt hơn trong tập nhỏ; cần nhiều danh tính/session/camera hơn trước khi đổi ngưỡng mặc định chung.

**19. FAR=0 có nghĩa không nhận nhầm?**  
Không. Pilot chỉ có tám impostor scores; FAR thay đổi theo bước 0,125. FAR=0 chỉ mô tả tập frozen test này.

**20. Embedding có phải dữ liệu vô danh không?**  
Không hoàn toàn; embedding vẫn dùng để liên kết quan sát với danh tính. Cần consent, phân quyền, retention, xóa đồng bộ ảnh/embedding và không chiếu dữ liệu chưa được phép.

**21. InsightFace có dùng thương mại được không?**  
Code và model có điều khoản khác nhau. Các pretrained model công khai như buffalo thường bị giới hạn nghiên cứu phi thương mại; triển khai thương mại cần kiểm tra/được cấp phép model phù hợp.

**22. Ultralytics YOLO có vấn đề license không?**  
Đề tài học thuật có thể dùng theo điều khoản nguồn mở phù hợp; nếu đóng gói sản phẩm proprietary/commercial cần tuân thủ AGPL hoặc mua giấy phép doanh nghiệp. Đây là ràng buộc triển khai, không phải metric kỹ thuật.

### E. Hành vi và Risk

**23. Intrusion trong sản phẩm là ROI hay line?**  
Trong phiên bản đóng băng: line hướng IN. Polygon method tồn tại nhưng không được gọi. Slide/demo phải dùng semantics này.

**24. Line crossing và intrusion có phải hai thuật toán độc lập?**  
Không. Chúng dùng chung geometry/counter. Intrusion là subset IN phát alert; runner quan sát cả IN/OUT để tạo hàng đánh giá line crossing.

**25. Tại sao dùng luật thay vì video transformer?**  
Luật nhẹ, giải thích được, cấu hình theo camera và phù hợp dữ liệu nhãn nhỏ. Nhược điểm là brittle theo góc nhìn/ngưỡng; mô hình học sâu chỉ hợp lý khi có dataset và tài nguyên đủ lớn.

**26. Theft dùng những bằng chứng gì?**  
Người lạ và phương tiện trong `asset_watch`, khoảng cách gần, thời gian gần bắt buộc, chuyển động phương tiện, cùng hướng và tiếp xúc cổ tay; cần đủ score/cue. Pilot cho thấy pose contact có thể kích hoạt sai nên phải siết bằng chứng chuyển động.

**27. Vì sao theft E2E passed nhưng frozen test F1=0?**  
E2E chỉ là một scenario thuận lợi. Frozen test có cả positive/negative và cửa sổ ground truth; nó phát hiện miss và false alert mà demo không thể hiện.

**28. Vì sao intrusion recall=1 nhưng precision thấp?**  
Sự kiện thật được phát hiện, nhưng cùng crossing sinh thêm năm event unmatched/duplicate. Vấn đề là dedup/state quanh line, không phải bỏ sót sự kiện mục tiêu.

**29. Vì sao line crossing latency 4,084 s?**  
TP hợp lệ xuất hiện muộn hơn eligible time; còn event sớm không khớp vẫn bị tính FP. Cần debounce/hysteresis và rà định nghĩa crossing/ground-truth window.

**30. Risk score 0,65 có ý nghĩa gì?**  
Chỉ là config tĩnh. Chưa có validation threshold; gate đang tắt. Không dùng 0,65 như bằng chứng khoa học.

**31. Làm thế nào đánh giá Risk đúng?**  
Gán nhãn từ video evidence, có cả đúng/sai và group theo người/scene/session; chuẩn hóa chỉ trên train; chọn threshold trên validation; khóa model/threshold; báo Accuracy, Precision, Recall, F1, ROC-AUC trên test độc lập.

**32. Vì sao không dùng lịch sử 36.674 alert để train ngay?**  
Đó là prediction, không phải ground truth; 18.218 label row đều trống, nhiều feature vector lặp và log trộn nhiều code version. Random event split sẽ rò rỉ.

**33. 22 feature có quá nhiều không?**  
Chúng thuộc năm nhóm rõ ràng và inference rất nhẹ. Cần regularization/ablation khi có dữ liệu để loại feature dư; hiện không được suy ra feature importance vì chưa có model hợp lệ.

### F. Thực nghiệm

**34. 206/206 chứng minh điều gì?**  
Chứng minh các contract phần mềm đã viết đang pass ở trạng thái code hiện tại. Không chứng minh accuracy ngoài đời hoặc không còn bug.

**35. Tại sao không báo mAP?**  
Không có ground-truth bounding box/class cho frame evaluation. Alert log không thể thay thế annotation detection.

**36. Tại sao không báo MOTA/IDF1/HOTA?**  
Không có trajectory ground truth theo frame. Đây là hạn chế và hướng phát triển MOT16/MOT17/benchmark nội bộ có annotation.

**37. Pilot một người có đáng tin không?**  
Đáng tin cho việc kiểm tra pipeline/freeze và phát hiện lỗi trong đúng điều kiện đã ghi; không đáng tin để suy rộng theo người hoặc bối cảnh. Nhóm chủ động gọi đây là deadline-safe pilot.

**38. Tại sao không tính điểm trung bình năm hành vi?**  
Mỗi hành vi chỉ có một event dương và một clip âm; gộp sẽ tạo cảm giác chắc chắn giả và trộn các semantics khác nhau.

**39. Vì sao development và test chưa đủ độc lập?**  
Tách session/repetition nhưng cùng P01 xuất hiện ở cả hai, nên không participant-disjoint. Nó giảm leakage theo clip nhưng không loại phụ thuộc theo người.

**40. Stability 8 giờ có đủ không?**  
Đủ để cung cấp bằng chứng vận hành bước đầu, không đủ để kết luận production dài ngày. Cần soak test dài hơn, RTSP thực, network fault injection và nhiều cấu hình.

**41. “Drop=0” có nghĩa không mất frame không?**  
Chỉ có nghĩa logger không ghi nhận drop theo định nghĩa hiện tại. Không chứng minh không có mất frame ở mọi tầng hoặc mọi điều kiện.

**42. RTX 5060 Ti GPU usage chỉ 13,8% mà sao không nhanh hơn?**  
Pipeline có CPU preprocessing, capture, serialized inference lock và giới hạn nhịp; GPU utilization trung bình thấp không có nghĩa không có burst hoặc bottleneck ngoài GPU. Cần profiler/batching để kết luận sâu hơn.

### G. Sản phẩm, bảo mật và triển khai

**43. Alert history có mất khi restart không?**  
Code hiện tại ghi SQLite và API ưu tiên đọc SQLite. RAM deque chỉ là fallback gần. Video evidence trước/sau alert mới là phần chưa hoàn thiện.

**44. Hệ thống có authentication không?**  
Có session login bảo vệ dashboard/API/MJPEG. Password mặc định phải đổi; session secret/token phải đưa qua biến môi trường hoặc secret management khi triển khai.

**45. RTSP credential có an toàn không?**  
Hiện được truyền cho OpenCV theo URL, nên cần tránh log/chiếu công khai, giới hạn quyền camera, tách mạng và dùng secret storage. Đây là điểm cần hardening khi production.

**46. Nếu queue đầy thì sao?**  
Queue có giới hạn và high-water warning; cần chính sách ưu tiên/drop rõ hơn cho production, đặc biệt cảnh báo fall không được để chìm sau alert mức thấp.

**47. Tại sao cần human-in-the-loop?**  
Các nhãn “đáng ngờ”, “trộm cắp” hoặc “ngã” có false positive/false negative và ảnh hưởng cá nhân. Hệ thống là công cụ hỗ trợ, người vận hành phải xem bằng chứng trước hành động.

**48. Sản phẩm đã sẵn sàng thương mại chưa?**  
Chưa. Cần license phù hợp, privacy/consent, threshold validation lớn hơn, persistent evidence, security hardening, benchmark, monitoring và quy trình vận hành/sự cố.

### H. Hướng phát triển và câu hỏi “nếu có thêm thời gian”

**49. Việc ưu tiên số một là gì?**  
Sửa lỗi đã quan sát: dedup intrusion, cue chuyển động theft, debounce/hysteresis line crossing; sau đó khóa version mới và đánh giá participant-disjoint.

**50. Ablation nên làm thế nào?**  
Giữ nguyên video/model/config/hardware; lần lượt tắt track memory, CMC, duplicate suppression; lặp tối thiểu ba lần; đo IDF1, ID switches, fragmentation, duplicate boxes, AI FPS và latency.

**51. Mở rộng hơn bốn camera như thế nào?**  
Batched inference, tách process, nhiều GPU hoặc inference server; trước hết profile CPU/lock/copy để không tối ưu theo cảm tính.

**52. Làm thế nào giảm false alarm có hệ thống?**  
Thu thập continuous-video GT có cả non-event, phân tích lỗi theo tầng, hiệu chỉnh rule, sau đó mới huấn luyện/calibrate Risk; không chỉ hạ hoặc nâng một threshold.

**53. Vì sao RTX 3050 bốn camera là 16,40 FPS ở bảng scaling nhưng 24,05 FPS ở bảng stability?**  
Hai số đến từ hai lần chạy khác nhau và hiện metadata lưu cạnh báo cáo chưa đủ để chứng minh chúng hoàn toàn cùng cấu hình. Không nên suy đoán nguyên nhân. Cần đối chiếu raw CSV, model, `frame_skip`, `ai_max_fps`, source và code hash; trước khi làm xong, dùng 16,40 cho phép scaling và dùng chuỗi 24,05 chỉ để phân tích drift/stability trong chính phiên tám giờ đó.

**54. Tại sao gọi sáu tầng nhưng sơ đồ có số box khác nhau?**  
Đó là khác mức trừu tượng, không phải hai pipeline khác nhau. Khi trình bày, nhóm gộp Detection/Tracking/Pose thành Vision AI và tách Learning/Risk thành một tầng; triển khai code tách các module chi tiết hơn. Cần dùng một sơ đồ duy nhất trên slide để tránh gây hiểu nhầm.

## 10. Cụm từ nên và không nên dùng

| Không nên nói | Nên nói |
|---|---|
| Hệ thống chính xác 100%. | 206/206 contract phần mềm đạt; accuracy thực tế cần ground truth riêng. |
| Theft detection hoạt động tốt. | Scenario E2E passed, nhưng frozen pilot có FN=1 và false alert=1. |
| Intrusion phát hiện người vào ROI. | Ở code freeze, intrusion phát khi confirmed person cắt line theo hướng IN. |
| Risk AI giúp giảm báo giả. | Hạ tầng Risk đã có, nhưng chưa đủ nhãn/model để xác thực mức giảm báo giả. |
| Ngưỡng 0,35 là tối ưu. | 0,35 tốt nhất trong ba operating point của pilot hai danh tính. |
| Hệ thống ổn định. | Không quan sát memory leak/drift/throttling trong hai phiên bốn camera × 8 giờ. |
| Chạy realtime bốn camera. | Aggregate AI FPS đạt 16,40/29,83 tùy GPU và cấu hình đã đo. |
| Không mất frame. | Logger ghi drop=0 trong hai phiên đã thử. |
| YOLO11 là mô hình tốt nhất/mới nhất. | YOLO11s là lựa chọn phù hợp với stack, phần cứng và mục tiêu của đề tài. |
| Paper và báo cáo có cùng benchmark. | Hai artifact khác model/cấu hình; không so trực tiếp. |

## 11. Chuẩn bị theo vai trò trong nhóm

Không được bịa phân công. Hai thành viên nên thống nhất trước:

- Ai chịu trách nhiệm kiến trúc/pipeline/tracking.
- Ai chịu trách nhiệm identity/behavior/experiments/UI.
- Mỗi người vẫn phải trả lời được bản tóm tắt, số liệu chính, giới hạn và semantics intrusion.
- Với câu hỏi về đóng góp cá nhân: nói rõ file/module/thí nghiệm đã làm, vấn đề đã giải quyết và cách kiểm chứng; không trả lời chung chung “cùng làm”.

## 12. Checklist học trong 48 giờ cuối

### Bắt buộc thuộc lòng

- Pipeline sáu tầng và lý do tách sync AI/async alert.
- Vì sao YOLO11s, ByteTrack, InsightFace, rules + logistic.
- Ba cải tiến tracking: lost-track memory, CMC, duplicate suppression.
- Semantics thật của intrusion/line crossing.
- 206 tests; hai phiên 8 giờ; hai bảng scaling.
- Bảng behavior pilot và ba ngưỡng face pilot.
- Risk chưa có model/nhãn; gate false.
- Năm hạn chế lớn và năm hướng phát triển.

### Nên tập nói thành tiếng

1. Pitch 45 giây.
2. Giải thích kiến trúc trong 60 giây không nhìn slide.
3. Giải thích một bảng metric trong 60 giây.
4. Trả lời “đóng góp mới là gì?” trong 30 giây.
5. Trả lời “tại sao kết quả theft kém?” mà không né tránh.
6. Trả lời “vì sao không có mAP/MOTA/Risk ROC-AUC?” bằng bằng chứng dữ liệu.
7. Trình bày demo bằng cả phương án live và video dự phòng.

## 13. Nguồn đối chiếu

### Nguồn nội bộ

- Báo cáo mới nhất được dùng để trích mục tiêu, kiến trúc, số liệu và giới hạn: `E:\SCT_Camera\.codex_addendum_baocao_20260821_01\BAOCAO-reference-snapshot.docx`.
- Phần bổ sung về ablation, validity, privacy và ethics: `E:\SCT_Camera\NOI_DUNG_BO_SUNG_BAO_CAO.docx`.
- Paper bản sửa cuối: `E:\SCT_Camera\outputs\paper_revision_19_08\final_render_v4\Paper_48_FDSE_Frozen_Pilot_Revised_v4.pdf`.
- Frozen behavior result: `E:\SCT_Camera\experiments\priority_19_08\behavior_evaluation\frozen_test\FROZEN_TEST_RESULTS.md`.
- Face pilot: `E:\SCT_Camera\outputs\face_pilot_20260820\FACE_PILOT_REPORT.md`.
- Tài liệu hệ thống: `E:\SCT_Camera\docs\system-documentation.md`.
- Product/README: `E:\SCT_Camera\PRODUCT.md`, `E:\SCT_Camera\README.md`.

### Nguồn học thuật/chính thức nên đọc lại

- Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/
- ByteTrack, ECCV 2022: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf
- ArcFace, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- HOTA, IJCV 2020: https://www.graphics.rwth-aachen.de/publication/00207/
- InsightFace licensing: https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md
- Ultralytics licensing: https://www.ultralytics.com/license

---

**Nguyên tắc cuối cùng:** câu trả lời mạnh nhất không phải câu trả lời nghe tự tin nhất, mà là câu trả lời phân biệt rõ “sản phẩm đã làm”, “bằng chứng đã đo”, “phạm vi kết luận” và “việc còn thiếu”.
