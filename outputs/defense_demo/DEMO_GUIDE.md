# Demo bảo vệ khóa luận — dùng ngay

## Trước khi vào phòng

- Chép cả thư mục này vào laptop và USB.
- Mở thử `SCT_Camera_Final_Defense_Demo.mp4` (hoặc nhấp `PLAY_DEMO.cmd`), bật toàn màn hình và tắt âm thanh thông báo của Windows.
- Để sẵn video ở cửa sổ bên cạnh PowerPoint; không cần Internet, camera hay Telegram.
- Nếu demo live chạy ổn thì trình bày live trước. Nếu lỗi quá 10 giây, chuyển ngay sang video.

## Câu chuyển sang video dự phòng

> Do điều kiện máy chiếu hoặc kết nối tại phòng, em xin chuyển sang video dự phòng. Video này dùng dữ liệu ghi sẵn nhưng được xử lý bằng đúng pipeline và cấu hình thí nghiệm của hệ thống, không phải video dựng kết quả thủ công.

## Lời nói theo video tổng hợp

- **Dashboard:** “Đây là giao diện hệ thống chạy local: đăng nhập, live wall nhiều camera, trạng thái xử lý, ROI, bộ đếm và lịch sử cảnh báo. Các kênh gửi cảnh báo ngoài được tắt khi ghi hình để tránh lộ thông tin cấu hình.”
- **Chuyển cảnh:** “Phần tiếp theo là replay offline ba tình huống bằng đúng pipeline và cấu hình thí nghiệm của sản phẩm.”
- **Line crossing:** “Khi tâm đối tượng đi qua đường ảo, hệ thống xác định hướng và cập nhật bộ đếm IN/OUT.”
- **Intrusion:** “Khi người đi vào vùng ROI được bảo vệ, hệ thống sinh cảnh báo xâm nhập và làm nổi bật vùng cùng đối tượng.”
- **Theft:** “Hệ thống kết hợp thời gian người ở gần phương tiện, chuyển động và tín hiệu pose/contact; khi điểm vượt ngưỡng thì sinh cảnh báo.”
- **Màn hình kết quả:** “Cả ba sự kiện đều được pipeline tái tạo; video chỉ minh họa luồng hoạt động, còn độ chính xác được báo cáo bằng phần đánh giá định lượng.”

## Nếu giảng viên hỏi

- **Có phải real-time không?** “Video này là replay offline để dự phòng; sản phẩm live xử lý cùng pipeline. Em không dùng thời gian chạy video dự phòng để tuyên bố FPS real-time.”
- **Cảnh báo có bị dựng không?** “Không. Script chạy lại đúng clip và cấu hình thí nghiệm; nếu không tái tạo được alert thì quá trình build sẽ báo lỗi và không chấp nhận video.”
- **Vì sao không gửi Telegram trong video?** “Để demo không phụ thuộc Internet và không lộ token. Bằng chứng Telegram đã có ở slide sản phẩm.”
- **Một video có chứng minh độ chính xác không?** “Không. Video chỉ minh họa luồng hoạt động; độ chính xác được báo cáo riêng bằng tập gán nhãn và các chỉ số định lượng.”

## Tuyệt đối tránh

- Không gọi video này là “camera live”.
- Không nói một video demo chứng minh hệ thống chính xác hoàn toàn.
- Không mở file cấu hình Telegram hoặc để token/webhook xuất hiện trên màn hình.
