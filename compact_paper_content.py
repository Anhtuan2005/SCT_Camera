from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "wp": WP, "a": A}
qn = lambda tag: f"{{{W}}}{tag}"


REPLACEMENTS = {
    "Ma và cộng sự [1] nhận thấy việc ByteTrack chia phát hiện thành hai nhóm độ tin cậy cao/thấp phụ thuộc vào một ngưỡng điều chỉnh thủ công và đặc thù cho từng tập dữ liệu; nhóm tác giả đề xuất ngưỡng thích nghi dựa trên điểm thay đổi dốc nhất của đường cong điểm tin cậy đã sắp xếp. ByteTrack [2] chỉ ra rằng việc loại các hộp có độ tin cậy thấp có thể làm mất các dương tính thật ở những đối tượng bị che khuất một phần. Chiến lược BYTE trước tiên ghép các hộp điểm cao, sau đó dùng các hộp điểm thấp chưa được ghép trong một lượt liên kết thứ hai. SORT [3] và DeepSORT [4] sử dụng bộ lọc Kalman [5] để dự đoán chuyển động và thuật toán Hungary [6] để phân công, trong khi DeepSORT bổ sung véc-tơ nhúng ngoại hình. SCT Camera sử dụng liên kết BYTE thông qua Ultralytics và thêm bộ nhớ tái liên kết vết đã mất để giảm phân mảnh danh tính sau khi vết đã bị loại khỏi bộ theo dõi nền.":
        "Ma và cộng sự [1] đề xuất ngưỡng thích nghi cho cách ByteTrack chia phát hiện tin cậy cao/thấp. ByteTrack [2] giữ các hộp điểm thấp để phục hồi dương tính thật bị che khuất bằng lượt liên kết thứ hai sau các hộp điểm cao. SORT [3] và DeepSORT [4] dùng bộ lọc Kalman [5] và thuật toán Hungary [6], trong đó DeepSORT thêm véc-tơ nhúng ngoại hình. SCT Camera dùng liên kết BYTE qua Ultralytics và bộ nhớ tái liên kết vết mất để giảm phân mảnh danh tính.",
    "Các bộ phát hiện dựa trên Transformer gần đây đã thu hẹp khoảng cách về khả năng xử lý thời gian thực. RT-DETR [7] giới thiệu bộ mã hóa lai hiệu quả và cơ chế chọn truy vấn có độ bất định tối thiểu để loại bỏ NMS; DEIM [8] tiếp tục tăng tốc hội tụ của DETR bằng ghép một-một dày và hàm mất mát có xét khả năng ghép. SCT Camera sử dụng YOLO11 [9] vì ngưỡng tin cậy theo từng lớp, biến thể ước lượng tư thế và khả năng xuất ONNX/TensorRT giúp đơn giản hóa việc tích hợp, dù giao diện bộ phát hiện vẫn có thể tiếp nhận các mô hình thuộc lớp RT-DETR/DEIM. Dòng YOLO bắt đầu với bộ phát hiện một lần của Redmon và cộng sự [10]; các thiết kế sau đó như YOLOX [11] sử dụng đầu phát hiện không neo và tách riêng các nhánh phân loại với hồi quy.":
        "RT-DETR [7] dùng bộ mã hóa lai và chọn truy vấn bất định tối thiểu để loại NMS; DEIM [8] tăng tốc hội tụ bằng ghép một-một dày và hàm mất mát có xét khả năng ghép. SCT Camera dùng YOLO11 [9] nhờ ngưỡng theo lớp, biến thể tư thế và khả năng xuất ONNX/TensorRT, nhưng giao diện vẫn hỗ trợ RT-DETR/DEIM. Dòng YOLO bắt đầu từ bộ phát hiện một lần [10]; YOLOX [11] dùng đầu không neo và tách nhánh phân loại/hồi quy.",
    "Các bộ phát hiện hai giai đoạn như Faster R-CNN vẫn có giá trị như một đường cơ sở về độ chính xác trong các miền bị ràng buộc. Divya Deepak và Krishna Bhat [12] cho thấy việc lựa chọn cẩn thận các siêu tham số gồm backbone, bộ giải, tốc độ học và ngưỡng phát hiện có thể nâng trung bình chính xác-thu hồi của Faster R-CNN khoảng 10 điểm phần trăm trong một bài toán phát hiện phương tiện. Kết quả này nhấn mạnh rằng chỉ lựa chọn bộ phát hiện là chưa đủ: siêu tham số và thiết kế toàn bộ quy trình có ảnh hưởng đáng kể đến độ chính xác khi triển khai. SCT Camera áp dụng nguyên tắc đó bằng ngưỡng tin cậy theo từng lớp và bộ lọc hình dạng hộp bao riêng cho người, thay vì dùng một ngưỡng tin cậy chung.":
        "Faster R-CNN vẫn là đường cơ sở độ chính xác hữu ích. Divya Deepak và Krishna Bhat [12] cho thấy tối ưu backbone, bộ giải, tốc độ học và ngưỡng phát hiện có thể tăng khoảng 10 điểm phần trăm trung bình chính xác-thu hồi trong phát hiện phương tiện. Vì siêu tham số và toàn bộ quy trình ảnh hưởng lớn đến triển khai, SCT Camera dùng ngưỡng tin cậy theo lớp và bộ lọc hình dạng hộp riêng cho người.",
    "SCT Camera giữ lại các luật dễ giải thích và cấu hình theo từng camera được mô tả trong Mục 3.5, đồng thời bổ sung hồi quy logistic [15] trên một véc-tơ đặc trưng thủ công 22 chiều để quyết định ứng viên do luật kích hoạt nào được nâng thành cảnh báo. Cách tiếp cận này đánh đổi sức mạnh biểu diễn để có tính minh bạch, yêu cầu ít dữ liệu và chi phí suy luận thấp. Ngược lại, Sultani và cộng sự [16] học cách xếp hạng trực tiếp các đoạn bình thường và bất thường từ video CCTV chưa cắt. Các mô hình bất thường học sâu như vậy có thể nắm bắt những mẫu mà luật thủ công bỏ sót, nhưng thường đòi hỏi tập video gán nhãn lớn hơn, đồng thời khó thích nghi và khó giải thích hơn tại từng địa điểm triển khai.":
        "SCT Camera kết hợp các luật theo camera ở Mục 3.5 với hồi quy logistic [15] trên véc-tơ 22 đặc trưng để chọn ứng viên cảnh báo, ưu tiên tính minh bạch, ít dữ liệu và chi phí thấp. Ngược lại, Sultani và cộng sự [16] học xếp hạng các đoạn CCTV bình thường/bất thường. Mô hình học sâu có thể phát hiện mẫu luật bỏ sót nhưng cần nhiều dữ liệu gán nhãn hơn, khó thích nghi và giải thích tại từng địa điểm.",
    "Hình 1 tóm tắt sáu giai đoạn xử lý, được thực thi trong một vòng lặp đồng bộ trên luồng riêng của từng camera và đưa dữ liệu vào hàng đợi sự kiện bất đồng bộ. Thu nhận đọc các khung BGR thô; Thị giác AI thực hiện phát hiện và theo dõi; Danh tính phân giải nhãn người; Phân tích đánh giá các luật hành vi trên hình học vùng và đường; Học chấm điểm rồi tùy chọn chặn cảnh báo; cuối cùng Web/Cảnh báo phát luồng video và phân phối thông báo.":
        "Hình 1 tóm tắt sáu giai đoạn trên luồng riêng của từng camera và hàng đợi sự kiện bất đồng bộ: Thu nhận đọc khung BGR; Thị giác AI phát hiện/theo dõi; Danh tính phân giải nhãn; Phân tích áp dụng luật hình học; Học chấm điểm và tùy chọn chặn cảnh báo; Web/Cảnh báo phát luồng và phân phối thông báo.",
    "Mỗi camera đang bật chạy trên một luồng quy trình chuyên dụng, đọc khung hình bằng OpenCV, áp dụng phép xoay tùy chọn rồi chuyển khung đến bộ phát hiện YOLOv11 dùng chung giữa các camera (mặc định là yolo11n.pt) thông qua một khóa suy luận tái nhập. Khâu phát hiện hỗ trợ ngưỡng tin cậy riêng theo lớp được cấu hình cho từng nơi triển khai thay vì một ngưỡng toàn cục. Hệ thống cũng loại các hộp người có tỷ lệ khung hoặc diện tích nằm ngoài giới hạn cấu hình, qua đó giảm phát hiện nhầm những cấu trúc thẳng đứng và mảnh như rèm cửa hoặc khung cửa thành người.":
        "Mỗi camera chạy trên một luồng riêng, đọc và tùy chọn xoay khung bằng OpenCV rồi dùng chung bộ phát hiện YOLOv11 (mặc định yolo11n.pt) qua khóa suy luận tái nhập. Ngưỡng tin cậy được cấu hình theo lớp. Các hộp người có tỷ lệ khung hoặc diện tích ngoài giới hạn bị loại để giảm nhầm rèm hay khung cửa thành người.",
    "Lớp theo dõi bọc ByteTrack của Ultralytics và bổ sung ba cơ chế. Thứ nhất, bộ nhớ đệm vết bị mất giữ lại tâm cuối cùng, tỷ lệ hộp bao và lớp của một vết trong một cửa sổ ngắn sau khi bộ theo dõi nền loại vết đó. Một phát hiện mới gần về không gian và phù hợp về kích thước sẽ được gắn lại vào ID cũ thay vì tạo ID mới, nhờ đó giảm phân mảnh danh tính do che khuất ngắn. Thứ hai, giai đoạn bù chuyển động camera tùy chọn dùng luồng quang thưa [17] để ổn định dự đoán chuyển động khi camera rung hoặc lia nhẹ. Một lớp bọc quanh hiện thực GMC nền bắt lỗi theo dõi đặc trưng và quay về phép biến đổi đơn vị thay vì lan truyền ngoại lệ. Thứ ba, bước loại bỏ trùng lặp sau liên kết loại các hộp cùng lớp bị trùng khi độ giao trên hợp (IoU) của chúng":
        "Lớp bọc ByteTrack bổ sung ba cơ chế. Bộ nhớ vết mất giữ tâm, tỷ lệ hộp và lớp trong một cửa sổ ngắn để gắn phát hiện phù hợp về không gian/kích thước vào ID cũ. Bù chuyển động camera tùy chọn dùng luồng quang thưa [17]; nếu theo dõi đặc trưng lỗi, lớp bọc GMC quay về phép biến đổi đơn vị. Cuối cùng, loại bỏ trùng lặp sau liên kết loại các hộp cùng lớp khi độ giao trên hợp (IoU)",
    "Một vết được gán nhãn known_person khi tham chiếu khớp tốt nhất vượt ngưỡng tương tự có thể cấu hình, thường nằm trong khoảng 0,35-0,45 tùy điều kiện ánh sáng của camera. Vết được gán stranger sau khi sử dụng hết ngân sách xác nhận gồm nhiều lần thử không khớp qua các góc xoay khung hình; trong những trường hợp còn lại, trạng thái là pending_person. Bộ nhớ không gian ngắn hạn cho phép một vết vừa được tạo kế thừa danh tính đã xác nhận gần đó mà không phải chạy lại so khớp khuôn mặt, qua đó giảm hiện tượng nhấp nháy nhãn khi ByteTrack tạm thời gán lại ID vết.":
        "Vết được gán known_person khi độ tương tự tốt nhất vượt ngưỡng cấu hình, thường 0,35-0,45 tùy ánh sáng; được gán stranger sau nhiều lần thử không khớp qua các góc xoay, và nếu chưa đủ bằng chứng thì giữ pending_person. Bộ nhớ không gian ngắn hạn cho phép vết mới kế thừa danh tính đã xác nhận gần đó, giảm nhấp nháy nhãn khi ByteTrack gán lại ID.",
    "Mô hình YOLOv11-Pose chỉ được gọi khi có ít nhất một đối tượng theo dõi thuộc lớp người và chỉ ở gần các vùng giám sát tài sản đã bật chấm điểm hành vi trộm cắp, bởi suy luận tư thế gần như làm tăng gấp đôi độ trễ AI trên mỗi khung. Mười bảy điểm mốc COCO phát hiện được ghép với các vết bằng IoU và được dùng để suy ra tín hiệu pose_push_contact, tức các điểm mốc cổ tay nằm trong một khoảng cách nhỏ so với hộp bao phương tiện. Đây là một trong nhiều hành vi độc lập đưa vào luật chấm điểm trộm cắp ở Mục 3.5.":
        "YOLOv11-Pose chỉ chạy khi có người gần vùng tài sản đã bật chấm điểm trộm cắp vì suy luận tư thế gần gấp đôi độ trễ AI. Mười bảy điểm mốc COCO được ghép với vết bằng IoU để suy ra pose_push_contact khi cổ tay ở gần hộp phương tiện; đây là một tín hiệu của luật ở Mục 3.5.",
    "Các vùng được biểu diễn bằng đa giác và kiểm tra bằng hàm điểm thuộc đa giác của OpenCV (Hình 2, bên trái). Các đường đếm dùng dấu của tích có hướng giữa hai vị trí tâm liên tiếp p1, p2 của một vết và hai đầu mút l1, l2 của đường để phát hiện sự kiện cắt đường cùng hướng di chuyển (Hình 2, bên phải):":
        "Vùng được kiểm tra bằng hàm điểm thuộc đa giác của OpenCV (Hình 2, trái). Đường đếm dùng dấu tích có hướng giữa hai tâm liên tiếp p1, p2 và hai đầu mút l1, l2 để phát hiện cắt đường cùng hướng di chuyển (Hình 2, phải):",
    "Một sự kiện cắt đường được phát ra khi side(p) đổi dấu giữa hai khung liên tiếp; hướng IN/OUT được xác định từ dấu của mức thay đổi. Trên hai nguyên thủy hình học này, bảy luật hành vi chạy ở mỗi khung. Xâm nhập phát cảnh báo một lần khi một người chiếm vùng hạn chế và dùng một khoảng đệm số khung trước khi xem vùng là trống trở lại. Lảng vảng duy trì bộ đếm thời gian lưu trú riêng cho từng vết và so sánh với ngưỡng cấu hình của từng vùng. Người lạ chỉ cảnh báo sau khi vết đã được xác nhận là stranger, không phải khi còn pending. Người lạ đáng ngờ bổ sung kiểm tra mẫu chuyển động, gồm đứng gần như bất động hoặc đi qua đi lại, trước khi cảnh báo việc người lạ hiện diện kéo dài. Giám sát tài sản theo dõi xe đạp, phương tiện và túi trong vùng quan sát, rồi cảnh báo nếu tài sản rời vùng hoặc biến mất trong khi một người không xác định vừa ở gần. Hành vi trộm cắp đáng ngờ tích lũy các tín hiệu độc lập gồm thời gian ở gần xe, số lượt đi qua lại hai bên, độ dịch chuyển của xe, hướng chuyển động tương quan và tiếp xúc theo tư thế; cảnh báo chỉ phát khi đủ số tín hiệu độc lập theo cấu hình cùng xuất hiện. Cuối cùng, Bộ đếm qua đường ghi nhận số lượt cắt đường theo cả hai hướng cho từng đường.":
        "Sự kiện cắt đường phát khi side(p) đổi dấu giữa hai khung; dấu mức thay đổi xác định hướng IN/OUT. Bảy luật chạy trên hai nguyên thủy hình học. Xâm nhập cảnh báo khi người vào vùng hạn chế; Lảng vảng so thời gian lưu trú với ngưỡng vùng. Người lạ chỉ cảnh báo ở trạng thái stranger; Người lạ đáng ngờ còn xét đứng lâu hoặc đi qua lại. Giám sát tài sản cảnh báo khi xe, phương tiện hay túi rời/biến mất sau khi người không xác định ở gần. Luật trộm cắp tích lũy thời gian ở gần, số lượt đi qua lại, dịch chuyển xe, hướng chuyển động tương quan và tiếp xúc tư thế, rồi cảnh báo khi đủ tín hiệu cấu hình. Bộ đếm qua đường ghi lượt cắt theo hai hướng.",
    "Mỗi ứng viên cảnh báo do luật kích hoạt được chuyển thành một véc-tơ số x gồm 22 chiều. Các đặc trưng bao gồm tỷ lệ thời gian, tỷ lệ độ dài quỹ đạo và dịch chuyển thuần được chuẩn hóa theo đường chéo khung, độ tin cậy đối tượng, tỷ lệ diện tích hộp bao, số lượt đi qua lại và các chỉ báo một-nóng về lớp đối tượng cũng như cấu hình vùng. Véc-tơ được ghi nối tiếp vào nhật ký sự kiện JSONL cùng một event_id ổn định. Người vận hành gán nhãn sự kiện sau đó thông qua tệp CSV hoặc một REST endpoint nhỏ (GET/POST /api/behavior-events). Với véc-tơ trọng số w và độ chệch b, mô hình ước lượng xác suất rủi ro bằng hàm logistic (sigmoid):":
        "Mỗi ứng viên được biểu diễn bằng véc-tơ x gồm 22 đặc trưng: tỷ lệ thời gian, độ dài quỹ đạo, dịch chuyển chuẩn hóa theo đường chéo khung, độ tin cậy, tỷ lệ diện tích hộp, số lượt đi qua lại và chỉ báo một-nóng về lớp/vùng. Véc-tơ cùng event_id được ghi vào JSONL; người vận hành gán nhãn qua CSV hoặc GET/POST /api/behavior-events. Với trọng số w và độ chệch b, xác suất rủi ro là:",
    "Tập lệnh độc lập scripts/train_behavior_classifier.py khớp w và b bằng cách tối thiểu hóa hàm mất mát entropy chéo nhị phân có điều chuẩn L2 trên N sự kiện đã gán nhãn, với yi thuộc {0,1}, sử dụng hạ gradient theo lô được hiện thực hoàn toàn bằng NumPy:":
        "scripts/train_behavior_classifier.py khớp w và b bằng hạ gradient theo lô trong NumPy, tối thiểu hóa entropy chéo nhị phân có điều chuẩn L2 trên N sự kiện gán nhãn, với yi thuộc {0,1}:",
    "Mô hình thu được, gồm tham số chuẩn hóa trung bình/độ lệch, véc-tơ trọng số và độ chệch, được nạp khi chạy và tự động nạp nóng lại mỗi khi thời gian sửa đổi tệp thay đổi. Theo mặc định, mô hình chỉ chú thích cảnh báo bằng risk_score. Người vận hành có thể bật thêm gate_alerts để loại cảnh báo dưới điểm tối thiểu τ, qua đó đánh đổi độ bao phủ để giảm tỷ lệ báo giả sau khi đã tích lũy đủ dữ liệu gán nhãn.":
        "Mô hình gồm tham số chuẩn hóa, trọng số và độ chệch; hệ thống nạp khi chạy và nạp nóng lại khi tệp đổi. Mặc định mô hình chỉ gắn risk_score; gate_alerts có thể loại cảnh báo dưới ngưỡng τ để giảm báo giả sau khi có đủ dữ liệu gán nhãn.",
    "Cảnh báo được đẩy từ các luồng quy trình vào một asyncio.Queue có giới hạn 1.000 phần tử và phát cảnh báo mức nước cao tại 800 phần tử, thông qua cầu nối call_soon_threadsafe an toàn giữa các luồng. Một tác vụ bất đồng bộ duy nhất lấy sự kiện khỏi hàng đợi, áp dụng thời gian chờ riêng theo tổ hợp camera, loại cảnh báo, vùng/đường/vết rồi mới phân phối đến các kênh đã cấu hình: Telegram Bot API, Discord webhook và/hoặc còi cục bộ có thời gian chờ độc lập, như minh họa ở Hình 4. Hai kênh nhắn tin sử dụng HTTP client bất đồng bộ lâu dài, thử lại theo cấp số nhân và tự động tạo lại client khi kết nối bị đặt lại. Ứng dụng FastAPI cung cấp luồng MJPEG multipart riêng cho từng camera, giới hạn tối đa mười người xem đồng thời; REST API hỗ trợ CRUD cho camera, vùng, đường và cài đặt; giao diện Jinja2 có trình chỉnh sửa tương tác dựa trên canvas để vẽ vùng đa giác và đường đếm trực tiếp trên luồng video.":
        "Các luồng đẩy cảnh báo qua call_soon_threadsafe vào asyncio.Queue giới hạn 1.000 phần tử, với cảnh báo mức nước cao tại 800. Một tác vụ bất đồng bộ áp dụng thời gian chờ theo camera, loại cảnh báo, vùng/đường/vết rồi phân phối qua Telegram Bot API, Discord webhook và/hoặc còi cục bộ (Hình 4). HTTP client dùng kết nối lâu dài, thử lại cấp số nhân và tự tạo lại khi bị đặt lại. FastAPI cung cấp MJPEG riêng cho từng camera, tối đa mười người xem; REST API CRUD camera, vùng, đường, cài đặt; giao diện Jinja2/canvas cho phép vẽ vùng và đường trên luồng video.",
    "Các phép đo được thu thập trên máy trạm Windows 10 có CPU 16 luồng logic, RAM 15,6 GB và GPU NVIDIA GeForce RTX 3050 Laptop 4 GB, chạy Python 3.10.11. Hệ thống dùng yolo11n.pt cho phát hiện và yolo11n-pose.pt cho tư thế, cả hai đều có kích thước đầu vào 640 px trên CUDA. Khối lượng công việc phát lại video cục bộ cố định 1280×720 people_720p25.mp4, giảm xuống chiều cao xử lý 480 px, trong 30 giây đo cho mỗi trường hợp sau năm khung khởi động. Trường hợp hai camera sử dụng hai luồng thu nhận/theo dõi độc lập. Các phép đo thông lượng và độ trễ dùng quá trình phát lại tệp có thể tái lập, còn nghiên cứu tình huống ở mức hành vi trong Mục 4.5 dùng luồng camera IMOU trực tiếp được xử lý trực tuyến vào ngày 12/7/2026.":
        "Phép đo chạy trên Windows 10, CPU 16 luồng, RAM 15,6 GB, NVIDIA GeForce RTX 3050 Laptop 4 GB và Python 3.10.11. yolo11n.pt và yolo11n-pose.pt dùng đầu vào 640 px trên CUDA. Video 1280×720 people_720p25.mp4 được giảm xuống cao 480 px và đo 30 giây sau năm khung khởi động; cấu hình hai camera dùng hai luồng thu nhận/theo dõi. Thông lượng/độ trễ dùng phát lại tái lập, còn Mục 4.5 xử lý trực tuyến camera IMOU ngày 12/7/2026.",
    "Bảng 2 phân rã tổng độ trễ AI theo từng giai đoạn trong trường hợp một camera. Phát hiện/theo dõi và tư thế chi phối chi phí gần như ngang nhau, mỗi giai đoạn khoảng 19-20 ms tại p50, trong khi gán nhãn danh tính và đánh giá luật hành vi cộng lại không đến 1,2 ms tại p50. Kết quả này xác nhận lớp phân tích dựa trên hình học gần như không tạo thêm chi phí đo được so với các giai đoạn mạng nơ-ron.":
        "Bảng 2 cho thấy ở một camera, phát hiện/theo dõi và tư thế cùng chi phối chi phí với khoảng 19-20 ms tại p50; danh tính và hành vi cộng lại dưới 1,2 ms. Vì vậy, phân tích hình học gần như không thêm chi phí so với các giai đoạn mạng nơ-ron.",
    "Độ trễ danh tính không phải là phép đo chuẩn nhận dạng khuôn mặt theo nghĩa chặt chẽ. Chính sách nguồn video cục bộ dùng trong phép đo này gán nhãn người mà không gọi toàn bộ đường so khớp khuôn mặt trực tiếp, vì vậy giá trị danh tính trong Bảng 2 phản ánh chi phí quản lý trạng thái hơn là chi phí suy luận InsightFace. Một phép đo riêng trên dữ liệu khuôn mặt có sự đồng thuận được dành cho công việc tương lai nếu thông lượng phân giải danh tính trở thành tiêu chí nghiệm thu.":
        "Độ trễ danh tính trong Bảng 2 phản ánh quản lý trạng thái, không phải phép đo đầy đủ của InsightFace, vì nguồn video cục bộ gán nhãn người mà không gọi toàn bộ đường so khớp khuôn mặt. Nếu thông lượng phân giải danh tính trở thành tiêu chí nghiệm thu, cần đo riêng trên dữ liệu khuôn mặt có sự đồng thuận.",
    "Hình 5 chuyển các kết quả dạng bảng thành một phép so sánh trực quan. Khi tăng từ một lên hai camera, tổng thông lượng tăng 17,9%, còn độ trễ AI p50 và p99 tăng lần lượt 82,3% và 61,1%. Phát hiện/theo dõi và tư thế chi phối độ trễ ở cấu hình một camera; danh tính và hành vi vẫn dưới 1,2 ms tại p50.":
        "Hình 5 cho thấy từ một lên hai camera, tổng thông lượng tăng 17,9%, còn độ trễ AI p50/p99 tăng 82,3%/61,1%. Phát hiện/theo dõi và tư thế chi phối; danh tính và hành vi vẫn dưới 1,2 ms tại p50.",
    "Tại thời điểm sửa đổi bài báo, pytest đã thực thi 135 ca trong 22,29 giây: 130 ca đạt và năm ca chưa đạt. Cả năm lỗi đều giới hạn trong các trường hợp phân giải danh tính liên quan đến việc giữ người ở trạng thái pending và từ chối kế thừa danh tính không phù hợp. Do đó, lần chạy hiện tại chứng minh phạm vi hồi quy rộng nhưng chưa phải một bộ kiểm thử sạch hoàn toàn. Một trăm ba mươi ca đạt bao phủ di trú và hoàn tác cơ sở dữ liệu, lọc theo ngưỡng và hình dạng của bộ phát hiện, cơ chế dự phòng CMC và loại bỏ trùng lặp của bộ theo dõi, nhịp xử lý quy trình, tiện ích vẽ và xác thực, cùng các luật hành vi. Ba kiểm thử trọng tâm về trộm cắp đều đạt trong 0,28 giây, xác nhận khả năng hiển thị điểm trực tiếp, vận hành khi không cấu hình vùng và loại bỏ trùng lặp cảnh báo sau khi ID vết phương tiện thay đổi.":
        "pytest thực thi 135 ca trong 22,29 giây: 130 ca đạt, năm ca chưa đạt; các lỗi chỉ thuộc phân giải danh tính khi giữ pending hoặc từ chối kế thừa sai. Các ca đạt bao phủ di trú/hoàn tác cơ sở dữ liệu, lọc bộ phát hiện, dự phòng CMC, loại bỏ trùng lặp, nhịp quy trình, tiện ích vẽ, xác thực và luật hành vi. Ba kiểm thử trộm cắp đạt trong 0,28 giây, xác nhận hiển thị điểm, vận hành không có vùng và khử cảnh báo trùng sau khi ID phương tiện đổi.",
    "Bộ phát hiện đánh giá độc lập từng cặp người lạ đã xác nhận-phương tiện. Năm tín hiệu nhị phân đóng góp một điểm cho mỗi tín hiệu: ở gần kéo dài (near_vehicle_duration, cấu hình bởi proximity_seconds), đi qua lại nhiều lần (pacing_near_vehicle, cấu hình bởi pacing_min_passes), phương tiện dịch chuyển (vehicle_started_moving), người và phương tiện chuyển động cùng hướng (moving_same_direction, dựa trên độ tương tự cosin của véc-tơ) và tiếp xúc theo tư thế (pose_push_contact). Một ứng viên đủ điều kiện cảnh báo khi điểm đạt score_threshold. Hiện thực còn yêu cầu tín hiệu thời gian ở gần và ít nhất một tín hiệu liên quan đến phương tiện, gồm dịch chuyển, cùng hướng hoặc tiếp xúc. Cấu hình được báo cáo sử dụng 10 giây ở gần, hai lượt đi qua lại, ngưỡng cosin 0,65 và ngưỡng điểm 2.":
        "Mỗi cặp người lạ đã xác nhận-phương tiện được chấm theo năm tín hiệu: near_vehicle_duration (proximity_seconds), pacing_near_vehicle (pacing_min_passes), vehicle_started_moving, moving_same_direction (độ tương tự cosin) và pose_push_contact. Ứng viên đạt score_threshold, đồng thời phải có thời gian ở gần và ít nhất một tín hiệu dịch chuyển, cùng hướng hoặc tiếp xúc. Cấu hình dùng 10 giây, hai lượt đi qua lại, ngưỡng cosin 0,65 và ngưỡng điểm 2.",
    "Kết hợp Hình 6(a) và Hình 6(b) cho thấy một cặp trên luồng trực tiếp tiến triển từ một tín hiệu đang hoạt động nhưng dưới ngưỡng lên năm tín hiệu đang hoạt động sau khi cảnh báo được kích hoạt. Nghiên cứu tình huống chức năng này chứng minh khả năng thực thi luật trực tuyến; độ chính xác, độ bao phủ, điểm F1 và tỷ lệ báo giả vẫn cần được đánh giá trên một tập nhiều đoạn video có gán nhãn.":
        "Hình 6(a-b) cho thấy cặp trực tiếp chuyển từ một lên năm tín hiệu và kích hoạt cảnh báo. Ca chức năng xác nhận luật chạy trực tuyến; độ chính xác, độ bao phủ, F1 và tỷ lệ báo giả vẫn cần đánh giá trên nhiều video gán nhãn.",
    "Ba khía cạnh của hệ thống hiện tại vẫn chủ ý chưa hoàn thiện so với trạng thái mục tiêu của kiến trúc. Thứ nhất, lịch sử cảnh báo và sự kiện hành vi được giữ trong các cấu trúc bộ nhớ có giới hạn, cụ thể là deque riêng theo camera, dù lược đồ SQLite cho cảnh báo, lần phân phối thông báo, sự kiện/nhãn hành vi và đoạn video đã được di trú và kiểm thử đơn vị. Nối AlertManager với lược đồ này để lưu bền vững thay vì mất lịch sử khi khởi động lại là bước tiếp theo cấp thiết nhất. Thứ hai, chức năng ghi đoạn video bằng chứng, dùng bộ đệm vòng trước/sau sự kiện được tham chiếu bởi các bảng video_clips và alert_clips, đã sẵn sàng ở mức lược đồ nhưng chưa được hiện thực. Thứ ba, đánh giá phát hiện té ngã có giám sát vẫn bị chặn ở giai đoạn chấp nhận dữ liệu. Công cụ đóng băng manifest yêu cầu tập kiểm thử tối thiểu 30 sự kiện té và 50 sự kiện không té, đồng thời không được trùng ID camera hoặc người giữa các tập; dữ liệu hiện tại chưa đạt bất kỳ ngưỡng nào. Hệ thống lúc chạy có luật tư thế heuristic cho sự kiện giống té ngã, nhưng thành phần này chỉ được xem là tính năng an toàn tạm thời và không được báo cáo như một mô hình phát hiện té ngã có giám sát đã được xác thực.":
        "Ba hạn chế chính vẫn tồn tại. Thứ nhất, lịch sử cảnh báo/sự kiện ở các deque giới hạn theo camera, dù lược đồ SQLite cho cảnh báo, phân phối, sự kiện/nhãn hành vi và video đã được di trú, kiểm thử; ưu tiên tiếp theo là nối AlertManager để lưu bền vững. Thứ hai, ghi video bằng chứng với bộ đệm vòng trước/sau sự kiện và các bảng video_clips, alert_clips mới dừng ở mức lược đồ. Thứ ba, đánh giá té ngã có giám sát bị chặn vì dữ liệu chưa đạt tối thiểu 30 sự kiện té và 50 không té, không trùng ID camera/người giữa các tập. Luật tư thế heuristic hiện chỉ là biện pháp an toàn tạm thời, không phải mô hình té ngã có giám sát đã xác thực.",
    "Về hiệu năng, việc dùng chung một khóa suy luận YOLO cho các camera khiến thông lượng tăng dưới tuyến tính theo số camera: hai camera đạt tổng 24,47 FPS so với 20,76 FPS của một camera, thay vì gần gấp đôi. Đây là sự đánh đổi có chủ ý khi chỉ có một GPU dùng chung trên phần cứng laptop phổ thông; thiết kế đa tiến trình hoặc suy luận theo lô có thể cải thiện khả năng mở rộng nhưng làm tăng chi phí kỹ thuật. Cuối cùng, mọi kết quả theo dõi trong bài báo đều lấy từ video thử nghiệm nội bộ thay vì một bộ chuẩn theo dõi công khai. Việc xác thực số lần chuyển đổi danh tính và các chỉ số MOTA/IDF1 trên một tập tiêu chuẩn như MOT16 [18] được dành cho công việc tiếp theo, để các phần bổ sung tái liên kết vết mất và CMC ở Mục 3.2 có thể được so sánh trực tiếp với các biến thể ByteTrack đã công bố.":
        "Khóa suy luận YOLO dùng chung làm thông lượng tăng dưới tuyến tính: hai camera đạt 24,47 FPS so với 20,76 FPS cho một camera. Đây là đánh đổi khi dùng chung GPU laptop; đa tiến trình hoặc suy luận theo lô có thể mở rộng tốt hơn nhưng tăng chi phí kỹ thuật. Kết quả theo dõi hiện dùng video nội bộ; công việc tiếp theo sẽ đánh giá chuyển đổi danh tính và MOTA/IDF1 trên MOT16 [18] để so sánh tái liên kết vết mất và CMC ở Mục 3.2 với các biến thể ByteTrack công bố.",
    "Bài báo đã trình bày SCT Camera, một quy trình phân tích video gần thời gian thực gồm sáu giai đoạn, kết hợp phát hiện YOLOv11; bộ theo dõi dựa trên ByteTrack có bổ sung tái liên kết vết mất và bù chuyển động tùy chọn; phân giải danh tính bằng InsightFace; bộ máy hành vi đa luật dựa trên hình học; cùng cổng rủi ro hồi quy logistic có giám sát được huấn luyện từ phản hồi của người vận hành. Trên GPU laptop phổ thông, toàn bộ quy trình duy trì tổng thông lượng gần thời gian thực từ 20,76 đến 24,47 FPS tùy số camera, trong khi chi phí phân tích không đáng kể so với suy luận mạng nơ-ron. Hệ thống được đánh giá cùng một lần chạy hồi quy tự động gồm 135 ca, trong đó hiện có 130 ca đạt. Bài báo đã xác định và báo cáo minh bạch các khoảng trống cụ thể gồm lưu bền vững lịch sử cảnh báo, ghi video bằng chứng và tập dữ liệu đánh giá phát hiện té ngã có giám sát đang bị chặn; đây là các ưu tiên cho công việc tiếp theo. Các hướng mở rộng khác gồm thử nghiệm bộ phát hiện dựa trên Transformer như RT-DETR hoặc DEIM khi ngân sách GPU cho phép, và mở rộng mô hình rủi ro vượt ra ngoài hồi quy logistic khi tích lũy thêm phản hồi có gán nhãn từ người vận hành.":
        "Bài báo trình bày SCT Camera, quy trình sáu giai đoạn kết hợp YOLOv11, ByteTrack với tái liên kết vết mất và bù chuyển động, InsightFace, luật hành vi hình học và cổng rủi ro hồi quy logistic học từ phản hồi người vận hành. Trên GPU laptop, hệ thống đạt 20,76-24,47 FPS tùy số camera; 130 trong 135 ca hồi quy đạt, trong khi chi phí phân tích nhỏ so với suy luận mạng nơ-ron. Các ưu tiên tiếp theo là lưu bền vững cảnh báo, ghi video bằng chứng và hoàn thiện dữ liệu té ngã; các hướng mở rộng gồm RT-DETR/DEIM khi GPU cho phép và mô hình rủi ro mạnh hơn khi có thêm nhãn.",
    "Trong lịch sử, giám sát video tại hộ gia đình và doanh nghiệp nhỏ chủ yếu dựa vào việc con người liên tục theo dõi luồng trực tiếp hoặc các cơ chế ghi hình kích hoạt theo chuyển động còn đơn giản. Cả hai phương án đều khó mở rộng: giám sát thủ công tốn kém và dễ sai sót trong những ca làm việc kéo dài, còn bộ kích hoạt chuyển động tạo ra khối lượng lớn cảnh báo ít giá trị do cây lá bị gió lay, đèn xe đi ngang hoặc vật nuôi, từ đó nhanh chóng gây mệt mỏi vì cảnh báo. Những tiến bộ gần đây về phát hiện đối tượng thời gian thực và theo dõi đa đối tượng cho phép chuyển trọng tâm phân tích từ “có vật gì chuyển động hay không” sang “ai hoặc vật gì đang xuất hiện, ở đâu, trong bao lâu và mẫu hành vi đó có đáng để nâng mức cảnh báo hay không”.":
        "Giám sát video gia đình và doanh nghiệp nhỏ thường dựa vào theo dõi thủ công hoặc ghi hình kích hoạt theo chuyển động. Cách thứ nhất tốn kém, dễ sai sót trong ca dài; cách thứ hai tạo nhiều cảnh báo ít giá trị từ cây lay, đèn xe hay vật nuôi. Phát hiện thời gian thực và theo dõi đa đối tượng cho phép xác định đối tượng, vị trí, thời gian xuất hiện và mức đáng ngờ của hành vi thay vì chỉ phát hiện chuyển động.",
    "Tuy nhiên, để triển khai sự chuyển đổi này trong thực tế trên phần cứng biên có tài nguyên khiêm tốn, hệ thống phải đồng thời giải quyết một số vấn đề: (i) phát hiện và theo dõi phải chạy với tốc độ khung hình ổn định mà không làm vòng lặp hiển thị video thiếu tài nguyên; (ii) danh tính của các vết theo dõi phải tồn tại qua những khoảng che khuất ngắn mà không bị phân mảnh thành ID mới; (iii) các ngữ nghĩa hành vi như xâm nhập, lảng vảng hay tiếp cận tài sản một cách đáng ngờ phải được biểu đạt theo kiểu khai báo và cấu hình riêng cho từng camera, thay vì mã hóa cứng; và (iv) tỷ lệ báo giả của các kích hoạt dựa trên luật phải có khả năng giảm dần theo thời gian mà không làm mất tính giải thích của chính các luật đó. SCT Camera được xây dựng để giải quyết đồng thời cả bốn vấn đề thay vì xem chúng là các thành phần tách biệt.":
        "Triển khai trên phần cứng biên đòi hỏi: (i) phát hiện/theo dõi ổn định mà không nghẽn hiển thị; (ii) duy trì danh tính qua che khuất ngắn; (iii) cấu hình hành vi như xâm nhập, lảng vảng hay tiếp cận tài sản theo từng camera; và (iv) giảm báo giả nhưng giữ tính giải thích. SCT Camera tích hợp cả bốn yêu cầu.",
    "(1) Một kiến trúc quy trình sáu giai đoạn gồm Thu nhận, Thị giác AI (Phát hiện + Theo dõi), Danh tính, Phân tích, Học/Cổng rủi ro và Web/Cảnh báo. Kiến trúc này tách vòng lặp hiển thị video khỏi nhịp suy luận AI, nhờ đó hình ảnh tiếp tục được hiển thị theo tốc độ khung hình thu nhận trong khi suy luận chạy ở tốc độ thấp hơn và có thể cấu hình bằng frame_skip, ai_max_fps trên một luồng xử lý riêng cho từng camera.":
        "(1) Kiến trúc sáu giai đoạn Thu nhận, Thị giác AI, Danh tính, Phân tích, Học/Cổng rủi ro và Web/Cảnh báo, tách hiển thị khỏi nhịp suy luận. frame_skip và ai_max_fps cấu hình tốc độ AI trên luồng riêng cho từng camera.",
    "(2) Một lớp theo dõi bọc quanh hiện thực ByteTrack của Ultralytics [2, 11] với hai phần bổ sung không có trong chiến lược liên kết BYTE nguyên bản. Phần thứ nhất là bộ nhớ đệm tái nhận dạng vết bị mất ngắn hạn, cho phép gắn lại ID cũ cho một phát hiện phù hợp về vị trí không gian và kích thước sau khi bộ theo dõi nền đã loại vết đó. Phần thứ hai là giai đoạn bù chuyển động camera (CMC) tùy chọn dùng luồng quang thưa, kèm cơ chế dự phòng an toàn quay về phép biến đổi đơn vị khi việc theo dõi đặc trưng thất bại.":
        "(2) Lớp bọc ByteTrack của Ultralytics [2, 11] bổ sung bộ nhớ tái liên kết vết mất ngắn hạn theo vị trí/kích thước để khôi phục ID, cùng CMC tùy chọn dùng luồng quang thưa và dự phòng về phép biến đổi đơn vị khi theo dõi đặc trưng lỗi.",
    "(3) Một bộ máy hành vi ưu tiên hình học, được xây dựng trên hai nguyên thủy: xác định điểm thuộc vùng đa giác và phát hiện cắt đường bằng tích có hướng. Từ hai nguyên thủy này, hệ thống kết hợp bảy luật hành vi độc lập và có thể cấu hình theo từng camera: Xâm nhập, Lảng vảng, Phát hiện người lạ, Theo dõi người lạ đáng ngờ, Giám sát tài sản (bị di chuyển hoặc biến mất), Chấm điểm hành vi trộm cắp đáng ngờ và Đếm qua đường có hướng.":
        "(3) Bộ máy hành vi ưu tiên hình học, dùng điểm-trong-đa-giác và cắt đường bằng tích có hướng để triển khai bảy luật theo camera: Xâm nhập, Lảng vảng, Phát hiện người lạ, Theo dõi người lạ đáng ngờ, Giám sát tài sản, Chấm điểm trộm cắp và Đếm qua đường có hướng.",
    "(4) Một cổng rủi ro hành vi có giám sát: bộ phân loại hồi quy logistic gồm 22 đặc trưng, được huấn luyện từ nhật ký sự kiện JSONL do người vận hành gán nhãn bằng tập lệnh scripts/train_behavior_classifier.py. Mô hình gắn risk_score cho mỗi cảnh báo do luật kích hoạt và, nếu được bật, sẽ loại các cảnh báo dưới một ngưỡng cấu hình mà không yêu cầu thay đổi logic của các luật nền.":
        "(4) Cổng rủi ro hồi quy logistic 22 đặc trưng, huấn luyện từ JSONL do người vận hành gán nhãn bằng scripts/train_behavior_classifier.py, gắn risk_score và tùy chọn lọc cảnh báo dưới ngưỡng mà không đổi logic luật.",
    "(5) Một phép đánh giá thực nghiệm có thể tái lập cho toàn bộ quy trình, bao gồm phát hiện/theo dõi, tư thế, danh tính và hành vi, ở cấu hình đồng thời một và hai camera trên GPU laptop phổ thông; đi kèm là một lần chạy hồi quy tự động gồm 135 ca. Trong phiên bản hiện tại, 130 ca đạt và năm ca liên quan đến phân giải danh tính chưa đạt. Phạm vi kiểm thử đạt bao gồm cơ chế dự phòng CMC của bộ theo dõi, lọc kết quả phát hiện, di trú cơ sở dữ liệu, xác thực, tiện ích vẽ, chấm điểm trộm cắp và nhịp thực thi quy trình.":
        "(5) Đánh giá tái lập toàn bộ quy trình với một/hai camera trên GPU laptop và 135 ca hồi quy: 130 ca đạt, năm ca phân giải danh tính chưa đạt. Ca đạt bao phủ dự phòng CMC, lọc phát hiện, di trú cơ sở dữ liệu, xác thực, tiện ích vẽ, chấm điểm trộm cắp và nhịp quy trình.",
    "Nhận dạng khuôn mặt dựa trên véc-tơ nhúng là một phương án nhẹ hơn so với tái nhận dạng toàn thân khi mục tiêu là phân biệt thành viên hộ gia đình hoặc nhân viên đã biết với người lạ. SCT Camera sử dụng các véc-tơ nhúng InsightFace được huấn luyện bằng hàm mất mát ArcFace có biên góc cộng [13], thường trên các backbone mạng dư [14], rồi thực hiện so khớp bằng độ tương tự cosin trong mô hình ba trạng thái gồm pending, known và stranger để tránh phát cảnh báo người lạ quá sớm.":
        "InsightFace dùng véc-tơ nhúng ArcFace [13] trên backbone mạng dư [14] để so khớp cosin, phân biệt người đã biết với người lạ bằng ba trạng thái pending, known và stranger, tránh cảnh báo quá sớm.",
    "Bộ nhớ cư trú tăng từ mức nền khi khởi động lạnh lên cực đại 1.939,70 MB cho một camera và 2.097,41 MB cho hai camera (Hình 5d). Mức tăng 157,71 MB, tương đương 8,1%, phù hợp với việc nạp mô hình trễ và lưu đệm hơn là nhân đôi tuyến tính tài nguyên theo từng camera. RSS cực đại không tiếp tục tăng trong cửa sổ đo.":
        "RSS đạt cực đại 1.939,70 MB với một camera và 2.097,41 MB với hai camera (Hình 5d), tăng 157,71 MB (8,1%). Mức tăng phù hợp với nạp mô hình trễ và lưu đệm, không phải nhân đôi tuyến tính; RSS không tăng tiếp trong cửa sổ đo.",
    "Trong thí nghiệm ở mức hành vi, nhóm tác giả dàn dựng một tương tác người-xe máy và xử lý trực tuyến luồng camera IMOU trực tiếp vào ngày 12/7/2026. Hình 6 theo dõi cùng một cặp người-phương tiện (P#4 + V#1) từ trạng thái dưới ngưỡng đến trạng thái cảnh báo đã xác nhận.":
        "Thí nghiệm hành vi xử lý trực tuyến tương tác người-xe máy từ camera IMOU ngày 12/7/2026. Hình 6 theo dõi cặp P#4 + V#1 từ dưới ngưỡng đến cảnh báo.",
    "Hình 6(a) cho thấy P#4 + V#1 trước khi kích hoạt: thời gian ở gần 9,1/10 giây, 1/2 lượt đi qua lại, tiếp xúc là tín hiệu duy nhất đang bật và điểm số là 1/2.":
        "Hình 6(a): P#4 + V#1 ở gần 9,1/10 giây, đi qua lại 1/2, chỉ có tín hiệu tiếp xúc và đạt 1/2.",
    "Hình 6(b) cho thấy cùng cặp sau khi kích hoạt: thời gian ở gần 37,5/10 giây, 2/2 lượt đi qua lại, các tín hiệu same-dir, near, pacing, contact và moved đều được bật, đồng thời xuất hiện THEFT ALERT 5/2. ID cặp ổn định và dấu thời gian hiện tại của camera ghi nhận việc xử lý trực tuyến một kịch bản luồng trực tiếp liên tục.":
        "Hình 6(b): cùng cặp đạt 37,5/10 giây, đi qua lại 2/2; same-dir, near, pacing, contact và moved đều bật, phát THEFT ALERT 5/2. ID ổn định trong luồng trực tiếp.",
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def replace_paragraph_text(paragraph: etree._Element, replacement: str) -> None:
    ppr = paragraph.find(qn("pPr"))
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, qn("r"))
    text = etree.SubElement(run, qn("t"))
    text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text.text = replacement


def style_id(paragraph: etree._Element) -> str | None:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else None


def move_figure_four(body: etree._Element) -> None:
    children = list(body)
    caption = next(
        node
        for node in children
        if node.tag == qn("p")
        and style_id(node) == "TenHinh"
        and "AlertManager" in paragraph_text(node)
    )
    caption_index = children.index(caption)
    image = children[caption_index - 1]
    assert image.xpath(".//w:drawing", namespaces=NS)
    explanation = next(
        node
        for node in children[caption_index + 1 :]
        if node.tag == qn("p")
        and style_id(node) == "NoiDung"
        and "asyncio.Queue" in paragraph_text(node)
    )
    body.remove(image)
    body.remove(caption)
    insert_at = list(body).index(explanation) + 1
    body.insert(insert_at, image)
    body.insert(insert_at + 1, caption)


def scale_drawings(root: etree._Element, factor: float) -> int:
    if factor <= 0 or factor > 1:
        raise ValueError("Drawing scale factor must be in (0, 1]")
    for extent in root.xpath(".//wp:extent | .//a:xfrm/a:ext", namespaces=NS):
        for attribute in ("cx", "cy"):
            value = extent.get(attribute)
            if value is not None:
                extent.set(attribute, str(round(int(value) * factor)))
    return len(root.xpath(".//w:drawing", namespaces=NS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--move-figure-four", action="store_true")
    parser.add_argument("--drawing-scale-factor", type=float, default=1.0)
    args = parser.parse_args()

    with zipfile.ZipFile(args.input, "r") as source:
        infos = source.infolist()
        parts = {info.filename: source.read(info.filename) for info in infos}

    root = etree.fromstring(parts["word/document.xml"])
    body = root.find(qn("body"))
    assert body is not None
    found = {}
    for paragraph in root.xpath("//w:body//w:p", namespaces=NS):
        text = paragraph_text(paragraph)
        if text in REPLACEMENTS:
            found[text] = found.get(text, 0) + 1
            replace_paragraph_text(paragraph, REPLACEMENTS[text])

    missing = [text[:80] for text in REPLACEMENTS if found.get(text) != 1]
    if missing:
        raise RuntimeError(f"Replacement source paragraph missing/non-unique: {missing}")
    if args.move_figure_four:
        move_figure_four(body)
    drawings = scale_drawings(root, args.drawing_scale_factor)

    before_words = sum(len(text.split()) for text in REPLACEMENTS)
    after_words = sum(len(text.split()) for text in REPLACEMENTS.values())
    replacements = {
        "word/document.xml": etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as target:
        for info in infos:
            target.writestr(copy.copy(info), replacements.get(info.filename, parts[info.filename]))

    print(f"paragraphs_compacted={len(REPLACEMENTS)}")
    print(f"words_before={before_words}")
    print(f"words_after={after_words}")
    print(f"words_removed={before_words - after_words}")
    print(f"moved_figure_four={args.move_figure_four}")
    print(f"drawings_scaled={drawings}")
    print(f"drawing_scale_factor={args.drawing_scale_factor}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
