# SCT Camera Realtime Monitoring

Hệ thống giám sát video thời gian thực:

Camera(s) -> YOLOv11 Detection -> ByteTrack Tracking -> Behavior Analysis -> Telegram Alert -> FastAPI Dashboard

## Tính năng chính

- YOLOv11 qua `ultralytics`, model mặc định `yolo11n.pt`.
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

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Cấu hình

Global settings nằm ở:

```text
config/settings.yaml
```

Các mục quan trọng:

- `telegram.chat_id`: điền group chat ID, thường có dạng số âm như `-1001234567890`.
- `detection.model`: `yolo11n.pt` nhanh hơn và đang dùng cho RTSP realtime; `yolo11s.pt` chính xác hơn nhưng chậm hơn.
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

YOLO model sẽ tự tải lần đầu từ Ultralytics hub. Nếu môi trường không có Internet, đặt sẵn file `yolo11n.pt` trong thư mục project hoặc đổi `detection.model` sang đường dẫn model local.

## Ghi chú vận hành

- Telegram token được đọc từ `config/settings.yaml`, không hardcode trong Python.
- Dashboard version 1 chưa có auth, chỉ nên chạy trong mạng nội bộ hoặc sau reverse proxy có bảo vệ.
- RTSP URL có username/password được truyền trực tiếp cho OpenCV.
- Ctrl+C sẽ kích hoạt FastAPI shutdown, dừng pipeline threads, release camera và dừng alert worker.
- Tất cả logs ghi ra console và `logs/sct_camera.log`.

## Video files and Discord alerts

- Camera `source` can be a webcam index, a local video path, or an RTSP URL.
- To replay a local video, set `source` in `config/cameras/*.yaml` to a path such as `E:\videos\sample.mp4`, or add it from `/settings`.
- Discord alerts use `discord.webhook_url` in `config/settings.yaml`.
- Each camera can choose `notification_channels`: `telegram`, `discord`, or both.
- Use `POST /api/settings/discord/test` or the Discord Test button in `/settings` to verify delivery.
