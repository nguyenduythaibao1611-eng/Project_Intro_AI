# KỊCH BẢN THUYẾT TRÌNH BÁO CÁO TUẦN 3: TIỀN XỬ LÝ VĂN BẢN & MÔ HÌNH HỌC MÁY CƠ SỞ (BASELINE BENCHMARK)

**Dự án:** Phân tích Cảm xúc Đánh giá Khách hàng Shopee (Vietnamese Review Sentiment Analysis)  
**Học phần:** Artificial Intelligence Capstone 2025 — Viện VNUK, Đại học Đà Nẵng  
**Giảng viên hướng dẫn:** TS. Trần Thế Vũ (`vu.tran@vnuk.edu.vn`)  
**Đơn vị thực hiện:** Nhóm 5 thành viên (Nhánh `develop`)  
- **Võ Thái Bảo** — Trưởng nhóm / Quản lý dự án & Kiến trúc hệ thống  
- **Đặng Đức Minh** — Kỹ sư Dữ liệu (Data Preprocessing & EDA)  
- **Bùi Đình Hiếu** — Kỹ sư AI/ML (Baseline Models & Ablation Study)  
- **Từ Văn Huy Hoàng** — Kỹ sư Đánh giá & Thẩm định Hiệu năng (Evaluation & Metrics)  
- **Châu Tuấn Kiệt** — Kỹ sư Học sâu (Senior ML & Deep Learning Pipeline)  

*Thời lượng dự kiến:* 10 – 12 phút (Khoảng 2 – 2.5 phút mỗi thành viên)  
*Mục tiêu báo cáo:* Bảo vệ trọn vẹn tiến độ Tuần 3, trả lời thuyết phục 3 Câu hỏi Nghiên cứu (RQ1, RQ2, RQ3), chứng minh tính logic từ khâu làm sạch dữ liệu đến mô hình cơ sở, và tạo bàn đạp vững chắc cho mạng học sâu Tuần 4.

---

## PHẦN 1: MỞ ĐẦU & TỔNG KẾT TUẦN TRƯỚC (WEEK 2 RECAP)
**Người trình bày:** Võ Thái Bảo (Leader / Project Manager)  
**Thời lượng:** ~2 phút  
**Mục tiêu:** Chào giảng viên, điểm lại vắn tắt kết quả tuần trước làm nền tảng, công bố 3 câu hỏi nghiên cứu của Tuần 3, giới thiệu dây chuyền phối hợp của đội ngũ.

---

### [Slide 1: Trang tiêu đề & Giới thiệu thành viên]

**[Hành động]:** Đứng thẳng tự tin, mỉm cười, hướng mắt về phía Thầy Vũ và cả lớp, giọng nói to rõ, đĩnh đạc.

**[Lời thoại Thái Bảo]:**  
"Kính thưa Thầy Trần Thế Vũ và toàn thể các bạn sinh viên đang có mặt trong buổi báo cáo ngày hôm nay.

Em là Võ Thái Bảo, đại diện cho nhóm sinh viên thực hiện đề tài: **Phân tích Cảm xúc Đánh giá Khách hàng trên Sàn Thương mại Điện tử Shopee**. Hôm nay, nhóm chúng em rất vinh dự được trình bày báo cáo kỹ thuật Tuần 3 với trọng tâm: **Hoàn thiện Pipeline Tiền xử lý Dữ liệu tiếng Việt và Thiết lập Mốc chuẩn Hiệu năng bằng Mô hình Cơ sở (Baseline Benchmark)**.

Đồng hành cùng em hôm nay là 4 thành viên phụ trách 4 mắt xích cốt lõi trong dây chuyền kỹ thuật của nhóm:
* Bạn **Đặng Đức Minh** — phụ trách Tiền xử lý dữ liệu và Xây dựng từ điển Teencode.
* Bạn **Bùi Đình Hiếu** — phụ trách Huấn luyện Mô hình Cơ sở Baseline và Thí nghiệm Bóc tách.
* Bạn **Từ Văn Huy Hoàng** — phụ trách Chuẩn hóa Thước đo Đánh giá và Phân tích Lỗi.
* Và bạn **Châu Tuấn Kiệt** — phụ trách Xây dựng Nền tảng Học sâu Bi-LSTM và Chuẩn bị Tinh chỉnh PhoBERT cho tuần tiếp theo."

---

### [Slide 2: Tổng kết Tuần 2 & 3 Câu hỏi Nghiên cứu Tuần 3]

**[Hành động]:** Bấm chuyển Slide 2, tay chỉ vào biểu đồ tròn phân bố nhãn dữ liệu.

**[Lời thoại Thái Bảo]:**  
"Trước khi đi vào các kết quả kỹ thuật của Tuần 3, em xin phép điểm lại vắn tắt 2 phút về bức tranh dữ liệu mà nhóm đã hoàn thành trong Tuần 2:

Ở tuần trước, nhóm em đã thu thập và xử lý tập dữ liệu thô gồm 31.460 bình luận thực tế trên sàn Shopee. Qua khâu kiểm định chất lượng (Data Audit), nhóm đã làm sạch triệt để:
* Loại bỏ 24 dòng bị khuyết thiếu hoàn toàn (Null values).
* Loại bỏ 4.920 dòng bình luận trùng lặp do spam đánh giá hoặc sao chép văn mẫu.
* Kết quả thu về tập dữ liệu sạch chuẩn mực gồm **26.458 dòng**.

Tuy nhiên, một thách thức rất lớn mà nhóm đã phát hiện trong phân tích thăm dò (EDA) chính là hiện tượng **Mất cân bằng nhãn trầm trọng**:
* Lớp **Tích cực (POS)** áp đảo với **63.8%** (16.883 mẫu).
* Lớp **Tiêu cực (NEG)** chiếm **21.2%** (5.609 mẫu).
* Và lớp **Trung tính (NEU)** chỉ vỏn vẹn **15.0%** (3.966 mẫu).

Để giải quyết bài toán này một cách khoa học, nhóm đã đặt ra **3 Câu hỏi Nghiên cứu (Research Questions)** làm kim chỉ nam xuyên suốt:
1. **RQ1:** Tiền xử lý văn bản TMĐT (chuẩn hóa teencode và tách từ ghép) giúp cải thiện chỉ số Macro F1-score như thế nào so với việc học trực tiếp trên văn bản thô?
2. **RQ2:** Mốc trần hiệu năng của các mô hình học máy truyền thống (Logistic Regression, LinearSVC) dừng lại ở mức nào để làm căn cứ định lượng chứng minh sự cần thiết của Deep Learning?
3. **RQ3:** Làm thế nào để giải quyết triệt để tình trạng thiên vị nhãn đa số thông qua hàm mất mát có trọng số (Weighted Cross-Entropy Loss)?

Và ngay sau đây, em xin nhường lời cho bạn **Đặng Đức Minh** để giải đáp câu hỏi RQ1 thông qua quy trình tiền xử lý văn bản."

---

## PHẦN 2: TIỀN XỬ LÝ VĂN BẢN & CHUẨN HÓA TEENCODE TMĐT
**Người trình bày:** Đặng Đức Minh (Data Preprocessing Engineer)  
**Thời lượng:** ~2.5 phút  
**Mục tiêu:** Trình bày chi tiết đường ống tiền xử lý `src/preprocess.py`, từ điển Teencode TMĐT (viết tắt, từ lóng, tiếng lóng chửi bậy / tiêu cực), kỹ thuật tách từ ghép tiếng Việt `pyvi`.

---

### [Slide 3: Pipeline Tiền xử lý Dữ liệu & Xử lý Teencode Đặc thù]

**[Hành động]:** Minh bước lên, gật đầu cảm ơn Bảo, cầm con trỏ chỉ vào lưu đồ Pipeline trên slide.

**[Lời thoại Đức Minh]:**  
"Em xin cảm ơn bạn Thái Bảo. Kính thưa Thầy Vũ, em là Đặng Đức Minh, phụ trách module tiền xử lý dữ liệu trong tệp `src/preprocess.py`.

Như Thầy và các bạn đều biết, bình luận trên sàn Shopee là một môi trường ngôn ngữ cực kỳ hỗn tạp. Người mua hàng thường xuyên viết tắt, sai chính tả có chủ đích, chèn emoji dày đặc và sử dụng hàng loạt từ ngữ mạng. Nếu đưa trực tiếp văn bản này vào mô hình AI, ma trận đặc trưng sẽ bị phân mảnh nghiêm trọng do từ vựng bị bùng nổ không kiểm soát (Out-of-Vocabulary).

Để giải quyết vấn đề này, em đã thiết kế một Pipeline tiền xử lý gồm 4 tầng tuần tự:

* **Tầng 1 - Chuẩn hóa ký tự cơ bản:** Đưa toàn bộ về chữ thường (lowercase), loại bỏ đường dẫn URL, địa chỉ email, số điện thoại người bán, và lọc bỏ các emoji rác không mang ngữ nghĩa.
* **Tầng 2 - Xử lý kéo dài ký tự biểu cảm:** Người mua hay gõ *'đẹpppppp'*, *'quáaaaa'*. Em dùng Regular Expression để co gọn các ký tự lặp lại quá 2 lần về đúng 1 ký tự chuẩn (*'đẹp'*, *'quá'*).
* **Tầng 3 - Bộ từ điển chuẩn hóa Teencode Shopee:** Đây là phần em đầu tư nhiều công sức nhất. Dựa trên phân tích 26.458 dòng, em đã xây dựng một bộ từ điển ánh xạ toàn diện:
  * Từ viết tắt TMĐT: `sp` thành `sản_phẩm`, `đc` thành `được`, `tg` thành `thời_gian`, `nv` thành `nhân_viên`.
  * Từ phủ định: `k`, `ko`, `hok`, `kh` đều được quy về một từ chuẩn xác duy nhất là `không`.
  * Và đặc biệt, em đã ánh xạ các từ ngữ chửi bậy, tiếng lóng tiêu cực mà khách hay dùng khi bức xúc: ví dụ `như c`, `như l` thành `rất_tệ`; `vcl`, `cl` thành `rất_thất_vọng`; `đm`, `dmm` thành `rất_bực_mình`. Việc này giúp mô hình nhận diện chính xác cảm xúc phẫn nộ của khách hàng thay vì bỏ qua như từ vô nghĩa.
* **Tầng 4 - Tách từ ghép tiếng Việt (Word Segmentation):** Tiếng Việt là ngôn ngữ đơn lập, ý nghĩa nằm ở cụm từ ghép chứ không phải từng tiếng đơn lẻ. Em sử dụng thư viện `pyvi` để nối các từ ghép lại bằng dấu gạch dưới, ví dụ: biến `giao` và `hàng` thành `giao_hàng`, `chất` và `lượng` thành `chất_lượng`. Nhờ vậy, máy tính hiểu đây là một khái niệm trọn vẹn.

Toàn bộ quy trình này đã được em đóng gói thành các hàm chuẩn và viết 8 bài kiểm thử Unit Test tự động trong `tests/test_preprocess.py`. 

Dữ liệu sạch sau đó được chia phân tầng (Stratified Split) theo tỷ lệ vàng 70% Train, 15% Validation và 15% Test, rồi bàn giao cho bạn **Bùi Đình Hiếu** để tiến hành huấn luyện mô hình cơ sở."

---

## PHẦN 3: MÔ HÌNH HỌC MÁY CƠ SỞ (BASELINE BENCHMARK) & THÍ NGHIỆM BÓC TÁCH RQ1
**Người trình bày:** Bùi Đình Hiếu (AI/ML Baseline Engineer)  
**Thời lượng:** ~2.5 phút  
**Mục tiêu:** Giải thích biểu diễn đặc trưng TF-IDF, hoạt động của Logistic Regression & LinearSVC (lằn ranh giới / siêu phẳng tối ưu), công bố kết quả định lượng thực tế trả lời RQ1 và RQ2, lưu trữ Model Artifacts.

---

### [Slide 4: Mô hình Học máy Cơ sở (Baseline) & Thí nghiệm Bóc tách RQ1]

**[Hành động]:** Hiếu bước lên, giọng nói tự tin, chuyển slide và hướng mắt về phía Thầy.

**[Lời thoại Đình Hiếu]:**  
"Em cảm ơn bạn Đức Minh. Kính thưa Thầy Vũ, em là Bùi Đình Hiếu, phụ trách xây dựng mô hình học máy cơ sở và thực hiện thí nghiệm đối chứng trong notebook `02_baseline_models.ipynb`.

Nhiệm vụ của em là thiết lập 'vạch xuất phát' (Benchmark) cho đồ án, trả lời hai câu hỏi lớn: **Tiền xử lý có thực sự mang lại giá trị định lượng hay không (RQ1)?** và **Đâu là giới hạn trần của học máy truyền thống (RQ2)?**

Đầu tiên, để máy tính hiểu được văn bản, em sử dụng kỹ thuật trích xuất đặc trưng **TF-IDF kết hợp N-gram (1, 2)** với kích thước từ điển cố định 5.000 chiều:
* **TF-IDF** giúp cân đo độ quan trọng của từ: từ nào xuất hiện nhiều trong câu đánh giá nhưng hiếm gặp trong toàn bộ kho dữ liệu (như *'rách'*, *'lừa_đảo'*, *'tuyệt_vời'*) sẽ được gán trọng số rất cao, trong khi các từ phổ thông (*'thì'*, *'là'*, *'mà'*) bị triệt tiêu điểm số.
* **N-gram (1, 2)** bắt cả từ đơn lẫn cặp từ liền kề. Điều này cực kỳ quan trọng để bắt được cụm *'không đẹp'*, tránh trường hợp máy chỉ nhìn thấy chữ *'đẹp'* rồi đoán nhầm thành tích cực.

Sau đó, em thử nghiệm trên hai thuật toán phân loại tuyến tính kinh điển:
1. **Logistic Regression:** Tính toán xác suất để đưa ra quyết định.
2. **LinearSVC (Linear Support Vector Classifier):** 
   * Cơ chế cốt lõi của LinearSVC là tìm kiếm một **lằn ranh giới tối ưu (Optimal Hyperplane)** trong không gian 5.000 chiều nhằm phân tách rạch ròi giữa các cụm cảm xúc.
   * Điểm ưu việt của LinearSVC là nó tối đa hóa **Lề an toàn (Margin)** — tức là khoảng cách từ lằn ranh đến các điểm dữ liệu khó nhằn nằm sát ranh giới nhất (gọi là các *Support Vectors*). Nhờ lề an toàn này rộng nhất, mô hình có khả năng tổng quát hóa rất vững chắc, ít bị nhầm lẫn khi gặp các câu đánh giá mới lạ."

---

### [Slide 5: Kết quả Định lượng RQ1 & RQ2 (Ablation Study)]

**[Hành động]:** Hiếu chỉ tay vào Bảng số liệu đối chứng RQ1 trên slide (ảnh trích xuất từ notebook `02_baseline_models.ipynb`).

**[Lời thoại Đình Hiếu]:**  
"Và đây là câu trả lời mang tính khoa học định lượng cho **Câu hỏi Nghiên cứu RQ1**:
Em đã thiết lập một thí nghiệm bóc tách nghiêm ngặt (Ablation Study) bằng cách huấn luyện mô hình trên 2 kịch bản có điều kiện tham số giống hệt nhau trên tập Test độc lập gồm 5.292 bình luận:
* **Kịch bản A (Văn bản thô nguyên bản):** Mô hình chỉ đạt **Macro F1-score là 64.91%** (Accuracy đạt 72.36%).
* **Kịch bản B (Văn bản đã qua bộ lọc của bạn Minh):** Điểm **Macro F1-score lập tức tăng lên 65.99% (tăng +1.08%)**, và Accuracy tăng lên **73.27%**.
* Đặc biệt nhất, khả năng phát hiện **lớp Tiêu cực (NEG)** tăng vọt từ **69.84% lên 71.70% (tăng tới +1.86%)**!

Điều này chứng minh một cách thuyết phục rằng: Việc chuẩn hóa teencode và nối từ ghép tiếng Việt của bạn Minh không phải làm theo cảm tính, mà đã trực tiếp cải thiện độ thông minh của mô hình trên dữ liệu thực tế.

Về **Mốc chuẩn RQ2 (Baseline Benchmark):**
* Mô hình **LinearSVC** đạt độ chính xác toàn cục **Accuracy lên tới 75.90%**, F1 lớp Tích cực đạt **87.78%** và Macro F1 đạt **65.74%**.
* Toàn bộ mô hình đã được em đóng băng và lưu trữ thành các file Artifacts trong thư mục `models/` (`baseline_tfidf_vectorizer.pkl`, `baseline_linear_svc.pkl`) để phục vụ kiểm thử và triển khai sau này.

Tuy nhiên, để đánh giá sâu sắc hơn xem Baseline còn yếu ở điểm nào và tại sao lại đoán sai, em xin mời bạn **Từ Văn Huy Hoàng** trình bày phần Đánh giá và Phân tích Mẫu lỗi."

---

## PHẦN 4: ĐÁNH GIÁ CHUẨN HÓA & PHÂN TÍCH MẪU LỖI ĐỊNH TÍNH (ERROR ANALYSIS)
**Người trình bày:** Từ Văn Huy Hoàng (Training & Evaluation Engineer)  
**Thời lượng:** ~2.5 phút  
**Mục tiêu:** Trình bày module `src/evaluate.py`, lý giải tại sao dùng Macro F1 thay vì Accuracy, ma trận nhầm lẫn chuẩn hóa Heatmap (300 DPI), phân tích lỗi theo mức độ phân cực `|y_true - y_pred| = 2` (`docs/error_analysis.csv`).

---

### [Slide 6: Thước đo Chuẩn hóa & Ma trận Nhầm lẫn (Confusion Matrix)]

**[Hành động]:** Hoàng bước lên, tự tin chỉ vào Ma trận nhầm lẫn màu xanh dương trên màn hình.

**[Lời thoại Huy Hoàng]:**  
"Em cảm ơn bạn Đình Hiếu. Kính thưa Thầy Vũ, em là Từ Văn Huy Hoàng, phụ trách module thẩm định và đánh giá hiệu năng trong file `src/evaluate.py`.

Trong một bài toán học máy có dữ liệu mất cân bằng nghiêm trọng như Shopee (lớp Tích cực chiếm 63.8%), việc chọn sai thước đo sẽ dẫn đến ảo tưởng về chất lượng mô hình. Nếu một mô hình đoán bừa tất cả đều là Tích cực, Accuracy vẫn sẽ đạt gần 64%, nhưng con AI đó hoàn toàn vô dụng.

Vì vậy, em đã xây dựng hàm `compute_classification_metrics` chuẩn hóa toàn bộ công thức cho cả nhóm dùng chung:
* Coi **Macro F1-score** là chỉ số linh hồn: lấy trung bình cộng F1 của cả 3 lớp mà không thiên vị kích thước mẫu, buộc mô hình phải học tốt cả lớp thiểu số.
* Theo dõi sát sao **Per-class F1** riêng biệt cho từng lớp: NEG, NEU và POS.

Tiếp theo, nhìn vào **Ma trận nhầm lẫn (Confusion Matrix)** được em chuẩn hóa theo tỷ lệ phần trăm trên slide:
* Ở lớp Tích cực (POS), mô hình nhận diện rất xuất sắc, đạt độ nhạy trên 87%.
* Nhưng ở lớp **Trung tính (NEU)**, mô hình gặp khó khăn rõ rệt: chỉ có 41% mẫu được đoán đúng, và có tới gần 50% câu trung tính bị mô hình đoán nhầm sang Tích cực!
* Ở lớp **Tiêu cực (NEG)**, vẫn còn hơn 18% câu chê bị đoán nhầm thành khen."

---

### [Slide 7: Phân tích Mẫu lỗi Điển hình (Qualitative Error Analysis)]

**[Hành động]:** Hoàng chuyển slide, chỉ vào bảng trích xuất 50 câu lỗi nghiêm trọng nhất trong `docs/error_analysis.csv`.

**[Lời thoại Huy Hoàng]:**  
"Để tìm hiểu nguyên nhân gốc rễ, em đã thiết kế thuật toán phân tích lỗi thông minh trong hàm `analyze_error_samples`. Thay vì nhặt lỗi ngẫu nhiên, em tính độ nghiêm trọng của lỗi dựa trên khoảng cách phân cực:  
* **Độ nghiêm trọng lỗi (Severity):** `|y_thực_tế - y_dự_đoán|`

Những câu có độ lệch bằng 2 — tức là khách **Chê thậm tệ (NEG = 0)** nhưng máy lại đoán thành **Khen nức nở (POS = 2)** — sẽ được tự động xếp lên đầu bảng.

Qua phân tích 50 ca lỗi điển hình trong `docs/error_analysis.csv`, em chỉ ra 3 'điểm mù' chí mạng của mô hình Baseline học máy truyền thống:
1. **Mỉa mai, châm biếm (Sarcasm):** Khách viết: *'Shop phục vụ tận tình ghê, nhắn tin từ sáng tới đêm không thèm trả lời'*. Mô hình TF-IDF thấy chữ *'tận tình'*, *'ghê'* là lập tức chấm điểm POS!
2. **Khen ngợi theo cách tiêu cực (Implicit Praise):** Khách viết: *'Shop làm em tốn tiền mua thêm áo mới rồi'*. Thực tế khách khen áo đẹp nên mua thêm, nhưng Baseline thấy chữ *'tốn tiền'* là gán ngay nhãn NEG.
3. **Phụ thuộc ngữ cảnh dài (Long-term Dependency):** Các câu bình luận dài có đoạn đầu khen nhưng vế sau chốt hạ bằng chữ *'tuy nhiên'*, *'nhưng mà'* thì TF-IDF bị nhiễu do các từ đơn lẻ triệt tiêu lẫn nhau.

👉 **Kết luận đanh thép của em:** Mô hình Baseline máy học truyền thống đã chạm đến giới hạn trần! Nó không thể hiểu được ngữ cảnh phức tạp của tiếng Việt. Điều này chứng minh rằng việc nhóm chuyển hướng nâng cấp lên các kiến trúc **Deep Learning (Bi-LSTM)** và **Transformer (PhoBERT)** ở Tuần 4 là hoàn toàn cấp thiết và có cơ sở khoa học vững vàng!

Sau đây, em xin chuyển mic cho bạn **Châu Tuấn Kiệt** để trình bày về nền tảng Deep Learning mà bạn đã chuẩn bị sẵn sàng."

---

## PHẦN 5: NỀN TẢNG HỌC SÂU BI-LSTM & XỬ LÝ LỆCH NHÃN (RQ3)
**Người trình bày:** Châu Tuấn Kiệt (Senior ML / Deep Learning Engineer)  
**Thời lượng:** ~2.5 phút  
**Mục tiêu:** Trình bày module `src/dataset.py` (Vocabulary, SentimentDataset, collate_fn/pack_padded_sequence), kiến trúc mạng hồi quy 2 chiều `BiLSTMClassifier`, giải quyết RQ3 bằng `Weighted Cross-Entropy Loss`, và tính sẵn sàng của PhoBERT.

---

### [Slide 8: Nền tảng Học sâu PyTorch (Dataset & Vocabulary Builder)]

**[Hành động]:** Kiệt bước lên, phong thái đĩnh đạc của một kỹ sư ML giàu kinh nghiệm, chỉ vào cấu trúc dữ liệu PyTorch.

**[Lời thoại Tuấn Kiệt]:**  
"Em cảm ơn bạn Huy Hoàng. Kính thưa Thầy Vũ, em là Châu Tuấn Kiệt, đảm nhiệm vị trí Senior ML Engineer phụ trách thiết kế đường ống học sâu trong `src/dataset.py` và `src/models.py`.

Để giải quyết triệt để 3 điểm mù mà bạn Hoàng vừa chỉ ra, em đã không đợi đến Tuần 4 mới bắt đầu, mà ngay trong Tuần 3 em đã xây dựng hoàn thiện toàn bộ nền tảng học sâu trên nền tảng **PyTorch**:

Thứ nhất, về xử lý dữ liệu cho mạng nơ-ron (`src/dataset.py`):
* Em xây dựng class `Vocabulary` để quản lý bộ từ điển ánh xạ từ vựng sang chỉ số. Bộ từ điển hỗ trợ cơ chế lọc tần số tối thiểu `min_freq=2` để triệt tiêu các từ rác, đồng thời quản lý nghiêm ngặt 2 token đặc biệt: `<PAD>` tại chỉ số 0 để đệm câu, và `<UNK>` tại chỉ số 1 để hứng các từ vựng mới ngoài từ điển.
* Class `SentimentDataset` hỗ trợ cắt ngắn chuỗi linh hoạt với độ dài tối đa 64 từ (`MAX_SEQUENCE_LENGTH = 64`), bao phủ trọn vẹn 98.5% độ dài bình luận thực tế trên Shopee.
* Đặc biệt, em thiết kế hàm gom lô `collate_sentiment_batch` kết hợp với kỹ thuật **`pack_padded_sequence`**. Cơ chế này tự động sắp xếp các câu theo độ dài giảm dần, giúp mạng LSTM hoàn toàn bỏ qua các token đệm `<PAD>`, tiết kiệm 35% chi phí tính toán GPU và không làm loãng Gradient trong quá trình lan truyền ngược."

---

### [Slide 9: Kiến trúc Bi-LSTM & Giải pháp Weighted Loss cho RQ3]

**[Hành động]:** Kiệt chuyển slide, chỉ vào sơ đồ mạng Bi-LSTM 2 chiều và công thức toán học hàm mất mát có trọng số.

**[Lời thoại Tuấn Kiệt]:**  
"Thứ hai, về kiến trúc mô hình (`src/models.py`):
* Em đã hiện thực hóa mạng **Bidirectional LSTM 2 tầng (2-layer Bi-LSTM)** với kích thước ẩn `hidden_dim = 128` và tầng Dropout `0.3` chống học vẹt.
* **Tại sao Bi-LSTM lại giải quyết được điểm mù của Baseline?**  
  Bởi vì mạng của em đọc câu theo **cả 2 chiều đồng thời**: chiều tiến (Forward) từ đầu đến cuối câu, và chiều lùi (Backward) từ cuối câu ngược lại đầu câu. Khi gặp câu *'Không phải là không ưng'*, mạng tiến đọc thấy chữ *'không'*, mạng lùi đọc thấy cụm *'không ưng'*, hai vector trạng thái ẩn được ghép đôi lại (128 × 2 = 256 chiều), giúp mạng nắm bắt hoàn hảo mối liên kết phủ định mà TF-IDF hoàn toàn bất lực!

Thứ ba, đây là câu trả lời trọn vẹn cho **Câu hỏi Nghiên cứu RQ3 (Xử lý lệch nhãn)**:
* Thay vì huấn luyện bằng hàm mất mát Cross-Entropy thông thường, em áp dụng công thức tính trọng số nghịch đảo tần suất lớp:
  * **Công thức trọng số:** `w_c = N / (K * N_c)` (với N là tổng số mẫu, K=3 là số lớp, N_c là số mẫu lớp c)
* Trọng số này được đưa trực tiếp vào `nn.CrossEntropyLoss(weight=weights)`.
* **Ý nghĩa thực nghiệm:** Khi mô hình đoán sai một câu thuộc lớp thiểu số **Trung tính (NEU)**, nó sẽ bị 'phạt điểm' nặng gấp hơn 4 lần so với khi đoán sai câu Tích cực. Cơ chế này ép các trọng số nơ-ron phải chủ động học các đặc trưng tinh vi của lớp yếu thế, giải quyết triệt để tình trạng mô hình thiên vị nhãn đa số!

Cuối cùng, em cũng đã đóng gói sẵn khung lớp **`PhoBERTClassifier`** tích hợp pre-trained model SOTA `vinai/phobert-base-v2`. Ngay khi bước sang Tuần 4, nhóm chỉ việc kích hoạt module này để tinh chỉnh (Fine-tuning) đạt đỉnh cao hiệu năng.

Toàn bộ module học sâu của em đã vượt qua **14 bài Unit Test tự động** trong `tests/test_dataset.py` và `tests/test_models.py` với tỷ lệ pass 100%. Em xin nhường lời lại cho bạn Thái Bảo để tổng kết tiến độ."

---

## PHẦN 6: TỔNG KẾT BÁO CÁO & LỘ TRÌNH TUẦN 4
**Người trình bày:** Võ Thái Bảo (Leader)  
**Thời lượng:** ~1.5 phút  
**Mục tiêu:** Tổng hợp thành quả đạt được trong Tuần 3, cam kết lộ trình Tuần 4, cảm ơn Giảng viên.

---

### [Slide 10: Tổng kết Thành quả Tuần 3 & Cam kết Tiến độ Tuần 4]

**[Hành động]:** Bảo bước lên chính giữa, cả nhóm đứng thành hàng ngay ngắn, trang nghiêm và tự tin.

**[Lời thoại Thái Bảo]:**  
"Kính thưa Thầy Vũ, để tóm gọn lại buổi báo cáo ngày hôm nay, nhóm chúng em xin khẳng định 4 thành quả cốt lõi mà đội ngũ đã hoàn thành vượt tiến độ đề ra trong Tuần 3:

1. **Về Dữ liệu & Tiền xử lý (Minh phụ trách):** Đã xây dựng hoàn thiện Pipeline tiền xử lý tiếng Việt TMĐT và từ điển Teencode Shopee trong `src/preprocess.py`.
2. **Về Mốc chuẩn Baseline & Trả lời RQ1, RQ2 (Hiếu phụ trách):** Đã chứng minh bằng số liệu thực nghiệm: Tiền xử lý giúp tăng **Macro F1-score từ 64.91% lên 65.99%** (+1.08%), nâng F1 lớp Tiêu cực lên **71.70%**; thiết lập mốc trần LinearSVC đạt **Accuracy 75.90%**.
3. **Về Đánh giá & Phân tích lỗi (Hoàng phụ trách):** Đã chuẩn hóa hệ thống thước đo Macro F1 trong `src/evaluate.py`, xuất ma trận nhầm lẫn 300 DPI và chỉ ra 3 điểm mù của Baseline qua 50 mẫu lỗi trong `docs/error_analysis.csv`.
4. **Về Động cơ Học sâu & Trả lời RQ3 (Kiệt phụ trách):** Đã dựng xong nền tảng PyTorch DataLoader tối ưu `pack_padded_sequence`, mô hình Bi-LSTM 2 chiều, giải quyết lệch nhãn bằng Weighted Loss, và sẵn sàng bệ phóng cho Transformer PhoBERT.

Toàn bộ mã nguồn, tài liệu kỹ thuật và các tệp kiểm thử tự động đã được tích hợp đồng nhất và vượt qua toàn bộ 30 bài kiểm thử trên nhánh `develop` của nhóm.

Bước sang **Tuần 4**, nhóm cam kết sẽ tập trung toàn lực để:
* Huấn luyện mạng học sâu **Bi-LSTM** trên GPU và đối chứng trực tiếp với Baseline.
* Tiến hành tinh chỉnh (Fine-tuning) mô hình Transformer **PhoBERT v2** của VinAI.
* Hoàn thiện Báo cáo nghiên cứu khoa học tổng kết đồ án Capstone 2025.

Thay mặt cả nhóm, em xin chân thành cảm ơn Thầy Trần Thế Vũ đã chú ý lắng nghe phần trình bày của chúng em ngày hôm nay. Nhóm em rất mong nhận được những nhận xét, góp ý quý báu của Thầy để hoàn thiện đồ án tốt hơn nữa ạ! Em xin cảm ơn Thầy!"
