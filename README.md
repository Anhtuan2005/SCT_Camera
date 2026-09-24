# SCT Camera Realtime Monitoring

Hệ thống phân tích video đa camera theo thời gian thực, kết hợp YOLOv11,
ByteTrack, OpenCV và FastAPI để phát hiện, theo dõi và cảnh báo sự kiện từ
webcam, video hoặc RTSP.

![SCT Camera realtime dashboard demo](media/sct-camera-demo.gif)

[Xem demo MP4 chất lượng cao](media/sct-camera-demo.mp4)

Demo 56 giây minh họa Live Wall đa camera, pose/activity classification,
fall detection và escalation, intrusion/ROI rules, asset-watch/theft scoring,
cùng luồng gửi cảnh báo bất đồng bộ.

## Kết quả nổi bật

- Xử lý một luồng 720p ở **26.5 FPS** trên NVIDIA RTX 3050 Laptop GPU.
- Đạt **27.6 FPS tổng** khi benchmark đồng thời bốn luồng video (**6.9 FPS/luồng** trung bình).
- **208 automated tests** đang pass.
- Dashboard realtime có pose/tracking overlay, ROI/line editor, alert history và cấu hình runtime.
- Behavior modules gồm fall/recovery, intrusion, loitering, stranger watch, asset missing/removed và experimental theft scoring.
- Behavior analytics là prototype thực nghiệm; kết quả pilot và giới hạn được đánh giá riêng, không coi là hệ thống an ninh production-ready.

Luồng xử lý chính:

Camera(s) -> YOLOv11 Detection -> ByteTrack Tracking -> Behavior Analysis -> Telegram Alert -> FastAPI Dashboard

Mỗi camera có capture/analysis worker và trạng thái tracker riêng. FastAPI cung cấp MJPEG stream, cấu hình ROI/line và lịch sử cảnh báo; notification worker gửi Telegram/Discord bất đồng bộ với cooldown, deduplication và retry. SQLite lưu alert history và cấu hình; ứng dụng chạy migration khi khởi động.

## Đánh giá và kiểm thử

- Benchmark dùng video local 1280 x 720, YOLO11s + pose, RTX 3050 Laptop GPU; mỗi mức 1/2/4 luồng chạy 3 lần, 15 giây/lần sau 30 frame warm-up. Kết quả trung bình: **26.5 FPS** (1 luồng), **27.6 FPS tổng / 6.9 FPS mỗi luồng** (4 luồng). Đây là throughput của pipeline benchmark, không phải accuracy của detector.
- Pilot có **40 clip** từ P01 và P02, mỗi người 20 clip chia development/test. Mỗi tập test giữ lại **10 clip**: 1 sự kiện dương và 1 clip âm cho mỗi hành vi. Hai kết quả dưới đây thuộc hai phiên bản đánh giá khác nhau (P01 v2, P02 v3), được trình bày riêng, không gộp thành một điểm chung.
- Bộ test tự động hiện chạy **208/208** bằng lệnh:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Giới hạn vào `tests` để tránh pytest thu thập bản clone kiểm tra nằm trong thư mục `tmp` cục bộ.

### P01 frozen test — 10 clip

| Hành vi | Đúng TP | Sai FP | Bỏ sót FN | F1 |
| --- | ---: | ---: | ---: | ---: |
| Intrusion | 1 | 5 | 0 | 0.29 |
| Loitering | 1 | 0 | 0 | 1.00 |
| Suspicious behavior | 1 | 0 | 0 | 1.00 |
| Theft | 0 | 1 | 1 | 0.00 |
| Line crossing | 1 | 2 | 0 | 0.50 |

### P02 test — 10 clip

| Hành vi | Đúng TP | Sai FP | Bỏ sót FN | F1 |
| --- | ---: | ---: | ---: | ---: |
| Intrusion | 0 | 5 | 1 | 0.00 |
| Loitering | 1 | 2 | 0 | 0.50 |
| Suspicious behavior | 1 | 1 | 0 | 0.67 |
| Theft | 0 | 1 | 1 | 0.00 |
| Line crossing | 0 | 5 | 1 | 0.00 |

TP là cảnh báo khớp sự kiện; FP là cảnh báo không khớp, gồm cả cảnh báo lặp trên clip dương; FN là sự kiện bị bỏ sót. Vì FP đếm **cảnh báo**, không đếm clip âm, một hành vi có thể có nhiều FP trên chỉ 2 clip. Mỗi F1 ở đây dựa trên đúng 1 sự kiện dương và 1 clip âm, nên không nên suy rộng thành hiệu năng thực tế. [CSV gốc rút gọn và phương pháp đọc số liệu](evidence/README.md) cho phép đối chiếu từng hàng; video và annotation chi tiết không nằm trong repo công khai.

## Tính năng chính

- YOLOv11 qua `ultralytics`; cấu hình public mặc định dùng `yolo11n.pt` để dễ chạy trên CPU.
- ByteTrack của Ultralytics với tracker state riêng cho từng camera.
- Camera motion compensation (CMC) bằng sparse optical flow để ổn định tracking khi camera rung/pan nhẹ.
- Multi-camera pipeline, mỗi camera chạy trên một thread riêng.
- FastAPI dashboard với MJPEG stream, ROI editor, line editor, settings và alert history.
- Intrusion detection, loitering detection, line crossing counter.
- Phát hiện người lạ khi identity đã thấy face đủ tốt nhưng không khớp người quen; người trong ROI intrusion ưu tiên báo intrusion.
- Nhận diện người quen bằng InsightFace face embedding, cache theo track ID.
- Loitering chỉ chạy trong ROI loitering/all với timer 30 giây mặc định hoặc threshold theo zone.
- Telegram alert async qua `httpx`, có cooldown, dedup, retry và history.
- Config YAML cho global settings và từng camera.

## Cài đặt trên Windows + CUDA

Tạo virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch CUDA trước. Ví dụ CUDA 12.4:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Cài các thư viện còn lại:

```powershell
pip install -r requirements.txt
```

Lần chạy đầu tiên, ứng dụng tự tạo cấu hình local an toàn từ
`config/settings.example.yaml` và `config/camera.example.yaml`. Các file cấu
hình thật được Git bỏ qua để không đưa token, mật khẩu hoặc địa chỉ camera lên
repository.

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Cấu hình

Global settings nằm ở:

```text
config/settings.yaml
```

File mẫu public nằm tại `config/settings.example.yaml`. Nếu cần khôi phục cấu
hình mặc định, xóa file local `config/settings.yaml` rồi chạy lại ứng dụng.

Các mục quan trọng:

- `telegram.chat_id`: điền group chat ID, thường có dạng số âm như `-1001234567890`.
- `detection.model`: cấu hình mẫu dùng `yolo11n.pt`; đổi sang `yolo11s.pt` để tăng độ chính xác nếu máy đủ mạnh.
- `detection.device`: dùng `cuda:0` nếu CUDA hoạt động, hoặc `cpu`.
- `detection.person_max_aspect_ratio`: bỏ bbox `person` quá mảnh/dài, giúp giảm rèm/cột bị nhận nhầm.
- `pipeline.frame_skip`: tăng lên để giảm tải GPU.
- `pipeline.camera_backend`: trên Windows nên để `msmf`; app tự bật workaround cho Logitech/UVC webcam.
- `tracking.camera_motion_compensation`: bật/tắt CMC; mặc định dùng `sparseOptFlow`, `downscale: 2`.
- `identity.similarity_threshold`: ngưỡng cosine của InsightFace; camera IMOU đang dùng `0.35` để nhận góc nghiêng/ngược sáng tốt hơn.

Camera config nằm trong:

```text
config/cameras/*.yaml
```

File mẫu public nằm tại `config/camera.example.yaml`; lần chạy đầu tiên file này
được sao chép thành `config/cameras/cam_01.yaml`.

`source` có thể là:

- `0` cho webcam mặc định.
- Đường dẫn file video như `E:\videos\sample.mp4`.
- RTSP URL như `rtsp://user:password@192.168.1.100:554/stream1`.

### Nhận diện người quen với InsightFace

Đặt ảnh khuôn mặt rõ nét vào `config/known_people`, rồi khai báo:

```yaml
identity:
  enabled: true
  model: buffalo_l
  device: auto
  similarity_threshold: 0.45
  known_persons:
  - name: Ba
    reference_images:
    - config/known_people/ba_front.jpg
    - config/known_people/ba_side.jpg
```

Model được tải vào `models/insightface` ở lần dùng đầu tiên. `onnxruntime`
trong requirements chạy CPU để không tranh GPU với YOLO. Pretrained model do
InsightFace cung cấp chỉ dành cho nghiên cứu phi thương mại; production thương
mại cần model có giấy phép phù hợp.

## Chạy hệ thống

```powershell
python main.py
```

Mở dashboard:

```text
http://localhost:8000
```

Các trang chính:

- `/` Dashboard nhiều camera.
- `/camera/{cam_id}` Stream lớn, ROI editor, line editor, alert history.
- `/settings` Telegram config, thresholds, thêm/xóa camera.

## Dashboard authentication

Dashboard, API và MJPEG stream được bảo vệ bằng session login. Mặc định trong
`config/settings.yaml`:

```yaml
web:
  auth:
    enabled: true
    username: admin
    password: change-me
```

Có thể override khi chạy demo mà không sửa file:

```powershell
$env:SCT_CAMERA_USERNAME="admin"
$env:SCT_CAMERA_PASSWORD="your-strong-password"
$env:SCT_CAMERA_SESSION_SECRET="long-random-session-secret"
python main.py
```

## API

- `GET /api/stream/{cam_id}` MJPEG stream.
- `GET /api/cameras` danh sách camera và status.
- `POST /api/cameras` thêm hoặc cập nhật camera.
- `DELETE /api/cameras/{cam_id}` xóa camera.
- `GET /api/cameras/{cam_id}/zones` lấy ROI zones.
- `POST /api/cameras/{cam_id}/zones` thêm hoặc cập nhật zone.
- `DELETE /api/cameras/{cam_id}/zones/{zone_id}` xóa zone.
- `GET /api/cameras/{cam_id}/lines` lấy counting lines.
- `POST /api/cameras/{cam_id}/lines` thêm hoặc cập nhật line.
- `DELETE /api/cameras/{cam_id}/lines/{line_id}` xóa line.
- `PUT /api/settings` cập nhật settings.
- `POST /api/settings/telegram/test` gửi test Telegram.
- `GET /api/alerts/{cam_id}` alert history.

## Test nhanh detection/tracking

Sau khi cài dependencies và có webcam/video:

```powershell
python main.py
```

Vào `http://localhost:8000/camera/cam_01`. Mặc định `cam_01` dùng source `0`. Nếu máy không có webcam, đổi `config/cameras/cam_01.yaml` sang file video hoặc RTSP URL.

YOLO model sẽ tự tải lần đầu từ Ultralytics hub. Nếu môi trường không có Internet, đặt sẵn model được khai báo trong `detection.model` vào thư mục project hoặc đổi mục này sang đường dẫn model local.

## Ghi chú vận hành

- Alert history và notification delivery được lưu bền vững trong SQLite tại `data/sct_camera.db`; app tự chạy migrations khi startup.
- Telegram token được đọc từ `config/settings.yaml`, không hardcode trong Python.
- Dashboard có session login; đổi password mặc định trước khi demo hoặc expose ra mạng.
- RTSP URL có username/password được truyền trực tiếp cho OpenCV.
- Ctrl+C sẽ kích hoạt FastAPI shutdown, dừng pipeline threads, release camera và dừng alert worker.
- Tất cả logs ghi ra console và `logs/sct_camera.log`.

## Video files and Discord alerts

- Camera `source` can be a webcam index, a local video path, or an RTSP URL.
- To replay a local video, set `source` in `config/cameras/*.yaml` to a path such as `E:\videos\sample.mp4`, or add it from `/settings`.
- Discord alerts use `discord.webhook_url` in `config/settings.yaml`.
- Each camera can choose `notification_channels`: `telegram`, `discord`, or both.
- Use `POST /api/settings/discord/test` or the Discord Test button in `/settings` to verify delivery.
