# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026.

**Runtime Colab:** T4 GPU; CUDA 13.0.

**Python / PyTorch / Ultralytics:** Python 3.13.16; PyTorch 2.11.0+cu130; Ultralytics 8.4.145.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`.

**Thay đổi so với notebook nguồn:** không


## 1. Phân loại ảnh

**Nguồn đối chiếu:** `classification_predictions.json`, `visuals/classification_top5.png`. Ảnh được chọn là `sample_id = traffic`, `rank = 1`.

1. **Record hạng 1:**
   - `class_id` = `468`
   - `class_name` = `cab`
   - `rank` = `1`
   - `score` = `0.510915`
   - `taxonomy_name` = `ImageNet-1K`

2. Record này nói về **toàn bộ bức ảnh**, không phải một vật thể riêng lẻ. Bài toán phân loại ảnh trả về các lớp có khả năng mô tả nội dung chung của ảnh. Ảnh `traffic` có nhiều phương tiện giao thông, nhưng mô hình phân loại vẫn đưa ra một danh sách lớp xếp hạng cho cả ảnh.

3. Danh sách tên lớp không phải do model tự nghĩ ra. Các lớp và mã lớp thuộc taxonomy đi kèm checkpoint, ở đây là `ImageNet-1K`.

4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name` để định danh lớp chính xác và tránh nhầm lẫn giữa các bộ taxonomy có thể dùng tên hoặc thứ tự lớp khác nhau. Tên lớp đơn lẻ chưa đủ để xác định duy nhất lớp trong mọi hệ thống.

5. Guideline cần quy định rõ mục tiêu phân loại và nguyên tắc chọn lớp đại diện khi ảnh có nhiều vật thể/chủ đề, ví dụ ưu tiên chủ thể chính, đối tượng chiếm diện tích lớn nhất hay chủ đề tổng thể. Quy tắc phải nhất quán giữa các annotator.

6. Không được lấy `score` làm ground truth vì score chỉ thể hiện mức độ tin cậy tương đối của mô hình đối với dự đoán. Nó không chứng minh nhãn đúng theo thực tế hoặc theo guideline. Ground truth phải do người gán nhãn xác định theo quy tắc đã thống nhất và được QC.

## 2. Phát hiện vật thể

**Nguồn đối chiếu:** `detection_predictions.json`, `visuals/detection_predictions.png`. Chỉ xét `sample_id = kitchen` (ảnh COCO `397133`, kích thước `640 × 427` pixel).

1. **Một record được chọn – đối tượng người ở phía bên phải ảnh:**
   - `class_name` = `person`
   - `score` = `0.912625`
   - `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
   - `bbox_width` = `113.58`
   - `bbox_height` = `279.68`
   - `taxonomy_name` = `COCO-80`

2. Box dùng hệ tọa độ pixel theo định dạng `xyxy`:
   - Góc trên-trái: `(x1, y1) = (385.33, 69.24)`.
   - Góc dưới-phải: `(x2, y2) = (498.92, 348.92)`.
   - Chiều rộng khoảng `113.58` pixel và chiều cao khoảng `279.68` pixel.
   - Như vậy box bao quanh người đứng ở khu vực bên phải ảnh, từ vị trí x khoảng 385 đến 499 và y khoảng 69 đến 349.

3. **So sánh hai ngưỡng trên ảnh `kitchen`:**
   - Ngưỡng `0.35`: có `11` prediction trong JSON.
   - Ngưỡng `0.60`: có `6` prediction (đếm các record có `score >= 0.60` trong danh sách prediction được lưu ở ngưỡng `0.35`).
   - Gói output chỉ lưu các prediction từ ngưỡng `0.35` trở lên, vì vậy không thể xác định số prediction ở ngưỡng `0.20` từ JSON hiện có.

4. Khi hạ ngưỡng, mô hình giữ lại nhiều hộp có score thấp hơn. Reviewer phải kiểm tra thêm các hộp dễ sai, hộp trùng lặp, hộp gán nhầm lớp, hộp quá rộng/hẹp và các đối tượng nhỏ hoặc bị che khuất; sau đó yêu cầu sửa hoặc loại bỏ nhãn không phù hợp theo guideline.

5. **Quy tắc box đề xuất:** Vẽ một hình chữ nhật bao sát phần nhìn thấy của từng đối tượng thuộc lớp cần gán nhãn; cạnh trái, trên, phải và dưới lần lượt chạm biên ngoài cùng của phần đối tượng nhìn thấy. Không tự suy diễn phần bị che khuất hoặc nằm ngoài khung ảnh; áp dụng cùng một quy tắc cho mọi ảnh.

6. Trên ảnh `kitchen`, có một người xuất hiện một phần ở mép trái ảnh. Box trong JSON có tọa độ x nhỏ nhất `0.12`, cho thấy đối tượng sát/cắt bởi biên ảnh. Với trường hợp vẫn nhận diện rõ là người, guideline có thể quy định chỉ đánh dấu phần nhìn thấy và box được phép chạm biên ảnh. Nếu phần còn lại bị che khiến không thể xác định chắc đối tượng hoặc biên cần vẽ, annotator phải escalation lên Lab Coach thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng

**Nguồn đối chiếu:** `segmentation_predictions.json`, `visuals/segmentation_prediction.png`. Chỉ xét `sample_id = kitchen`.

1. **Một record được chọn – người đứng bên phải ảnh:**
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.899318`
   - `polygon_point_count` = `348`
   - 5 điểm đầu của `polygon_xy`:
     `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. Box chỉ biểu diễn vùng chữ nhật bao quanh đối tượng. Polygon mô tả đường biên chi tiết hơn, giúp thể hiện hình dạng thực tế của người, kể cả những phần không lấp đầy toàn bộ hình chữ nhật.

3. `instance_id` dùng để phân biệt từng đối tượng cụ thể trong kết quả của ảnh, kể cả khi nhiều đối tượng có cùng `class_name`. Ví dụ hai chiếc bát có thể cùng lớp `bowl` nhưng có `instance_id` khác nhau. Nó không phải class ID vì class ID định danh loại đối tượng; nó cũng không phải tracking ID vì mã này không khẳng định theo dõi cùng một đối tượng qua nhiều khung hình/video.

4. **Quy tắc vẽ biên mask đề xuất:** Đi theo đường bao của phần đối tượng nhìn thấy, giữ mask sát biên và không bao gồm nền hoặc vật thể lân cận. Với phần bị che, chỉ vẽ phần nhìn thấy; không tự nối hoặc phỏng đoán phần khuất nếu guideline chưa quy định.

5. Trên ảnh `kitchen`, vùng mặt bàn có nhiều đồ vật đặt trên đó nên ranh giới giữa mặt bàn và các vật thể nhỏ ở khu vực tiếp xúc có thể khó xác định. Annotator cần áp dụng quy tắc biên mask thống nhất, tách riêng từng instance. Nếu hai vật chạm nhau hoặc biên bị che đến mức không thể xác định chắc chắn mask thuộc về vật nào, cần escalation để Lab Coach làm rõ guideline. Không nên tự vẽ theo suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

Chuỗi vòng đời: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định quan sát được trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh, kèm class ID/taxonomy theo guideline. | Ảnh `traffic` có nhiều loại phương tiện; model xếp `cab` hạng 1 với score `0.510915`, tiếp theo là `minibus` `0.164284`. Prediction là gợi ý, không tự động trở thành nhãn đúng. | Xác định chủ đề/đối tượng đại diện theo quy tắc chọn lớp; ghi nhãn ground truth và giải thích trường hợp đa chủ đề nếu cần. | Kiểm tra nhãn có đúng phạm vi toàn ảnh, đúng taxonomy và nhất quán với guideline; không dùng score làm căn cứ duy nhất. |
| Phát hiện vật thể | Một bounding box `xyxy` và class cho từng đối tượng nhìn thấy. | Ảnh `kitchen` có 11 prediction ở ngưỡng `0.35`; có người ở sát mép trái (`bbox` bắt đầu tại x=`0.12`) và nhiều đồ vật nhỏ gần nhau. | Vẽ box sát phần nhìn thấy, gán đúng lớp; áp dụng quy tắc cho đối tượng bị cắt mép/che khuất và escalation khi không chắc chắn. | Kiểm tra lớp, tọa độ, kích thước, box trùng/thiếu/thừa, box cắt sai đối tượng và ảnh hưởng của threshold. |
| Instance segmentation | Một mask/polygon riêng cho từng instance, kèm class và mã instance. | Ảnh `kitchen` có 11 mask; `kitchen-001` là `person` với 348 điểm polygon. Mặt bàn có nhiều vật thể đặt trên đó nên một số biên tiếp xúc có thể khó phân định. | Vẽ polygon theo biên phần nhìn thấy, tách riêng từng instance; hỏi Lab Coach khi ranh giới không thể xác định nhất quán. | Kiểm tra mask có bám sát biên, có lấn nền/vật khác, bỏ sót vùng nhìn thấy hoặc gộp nhầm instance hay không. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu ảnh và output thuộc phạm vi bài lab, không đưa thông tin cá nhân hoặc dữ liệu không được phép vào báo cáo/repository công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, dừng xử lý và báo cho Lab Coach/giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Đối chiếu với gói output đã cung cấp:

- [ ] `REPORT.md` (báo cáo này cần được đưa vào thư mục `report/` trên repository).
- [x] `day1_lab_outputs/classification_predictions.json` (có trong ZIP; hiện nằm ở thư mục gốc của ZIP).
- [x] `day1_lab_outputs/detection_predictions.json` (có trong ZIP; hiện nằm ở thư mục gốc của ZIP).
- [x] `day1_lab_outputs/segmentation_predictions.json` (có trong ZIP; hiện nằm ở thư mục gốc của ZIP).
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md` (có trong ZIP; hiện nằm ở thư mục gốc của ZIP).
- [x] `day1_lab_outputs/visuals/classification_top5.png` (có trong ZIP, trong `visuals/`).
- [x] `day1_lab_outputs/visuals/detection_predictions.png` (có trong ZIP, trong `visuals/`).
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png` (có trong ZIP, trong `visuals/`).
- [ ] Ô validation cuối notebook báo `PASS` (chưa có ảnh/chụp kết quả validation trong gói được gửi).
- [x] Không đưa họ tên, MSSV hoặc dữ liệu nhạy cảm vào nội dung báo cáo này.
- [ ] Đã nộp **link repository** trên LMS.
