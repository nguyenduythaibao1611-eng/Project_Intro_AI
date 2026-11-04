# TÀI LIỆU THAM KHẢO & CƠ SỞ KHOA HỌC (ACADEMIC REFERENCES)
**Dự án:** Phân tích Cảm xúc Đánh giá Khách hàng Shopee (Vietnamese Review Sentiment Analysis)  
**Học phần:** Artificial Intelligence Capstone 2025 — VNUK, Đại học Đà Nẵng  
**Giảng viên hướng dẫn:** TS. Trần Thế Vũ  
**Nhóm thực hiện:** Thái Bảo (Leader), Tuấn Kiệt, Đức Minh, Đình Hiếu, Huy Hoàng  

---

## 1. CÁC BÀI BÁO KHOA HỌC CỐT LÕI (CORE SCIENTIFIC PAPERS)

### [1] Bài báo gốc về VinAI PhoBERT (Bắt buộc trích dẫn cho Mô hình SOTA & RQ2)
* **Tiêu đề:** *PhoBERT: Pre-trained language models for Vietnamese*
* **Tác giả:** Dat Quoc Nguyen, Anh Tuan Nguyen (VinAI Research)
* **Hội nghị xuất bản:** Findings of the Association for Computational Linguistics: **EMNLP 2020** (pp. 1037–1042).
* **DOI / Link:** [https://doi.org/10.18653/v1/2020.findings-emnlp.92](https://doi.org/10.18653/v1/2020.findings-emnlp.92)
* **Mục đích trích dẫn trong đồ án:**
  * Minh chứng khoa học cho việc lựa chọn mô hình Transformer tối ưu cho tiếng Việt thay vì mBERT đa ngôn ngữ.
  * Căn cứ cho bước Tiền xử lý bắt buộc: tách từ tiếng Việt (Word Segmentation) bằng RDRSegmenter / PyVi trước khi đưa vào PhoBERT Tokenizer.
* **Mã BibTeX (Dùng cho LaTeX / Final Report):**
```bibtex
@inproceedings{nguyen2020phobert,
  title={{PhoBERT: Pre-trained language models for Vietnamese}},
  author={Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages={1037--1042},
  year={2020}
}
```

---

### [2] Bài báo chuẩn Benchmark Phân tích Cảm xúc Tiếng Việt (UIT-VSFC)
* **Tiêu đề:** *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis at the Sentence Level*
* **Tác giả:** Kiet Van Nguyen, Vu Duc Nguyen, Phu X. V. Nguyen, Tham T. H. Truong, Ngan Luu-Thuy Nguyen
* **Hội nghị xuất bản:** **IEEE 10th International Conference on Knowledge and Systems Engineering (KSE 2018)** (pp. 284–290).
* **DOI / Link:** [https://doi.org/10.1109/KSE.2018.8573337](https://doi.org/10.1109/KSE.2018.8573337)
* **Mục đích trích dẫn trong đồ án:**
  * Cơ sở phân loại bài toán cảm xúc tiếng Việt thành 3 nhãn chuẩn: Positive (Tích cực), Negative (Tiêu cực), Neutral (Trung tính).
  * Đối chứng quy mô dữ liệu: Bộ UIT-VSFC chuẩn có 16.000 câu $\rightarrow$ Bộ dữ liệu Shopee 26.500 câu của nhóm có quy mô tương đương và mang tính thực tiễn cao trong thương mại điện tử.
* **Mã BibTeX:**
```bibtex
@inproceedings{nguyen2018uit,
  title={{UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis at the Sentence Level}},
  author={Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu XV and Truong, Tham TH and Nguyen, Ngan Luu-Thuy},
  booktitle={2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
  pages={284--290},
  year={2018},
  organization={IEEE}
}
```

---

### [3] Bài báo về Mạng Nơ-ron Hai chiều (Bi-LSTM) cho Xử lý Chuỗi
* **Tiêu đề:** *Bidirectional LSTM-CRF Models for Sequence Tagging*
* **Tác giả:** Zhiheng Huang, Wei Xu, Kai Yu
* **Xuất bản:** arXiv:1508.01991 (2015).
* **Link:** [https://arxiv.org/abs/1508.01991](https://arxiv.org/abs/1508.01991)
* **Mục đích trích dẫn trong đồ án (Phục vụ RQ2):**
  * Biện luận cho việc chọn mạng Bi-LSTM thay vì RNN hay LSTM đơn chiều.
  * Giải thích cơ chế đọc hai chiều (Forward + Backward) giúp mô hình bắt trọn từ phủ định ("không", "chẳng", "chưa") đứng liền trước tính từ khen ngợi ("không đẹp", "chưa tốt").
* **Mã BibTeX:**
```bibtex
@article{huang2015bidirectional,
  title={{Bidirectional LSTM-CRF models for sequence tagging}},
  author={Huang, Zhiheng and Xu, Wei and Yu, Kai},
  journal={arXiv preprint arXiv:1508.01991},
  year={2015}
}
```

---

### [4] Nghiên cứu về Xử lý Mất cân bằng Nhãn (Class Imbalance & Loss Weighting)
* **Tiêu đề:** *Class-Balanced Loss Based on Effective Number of Samples*
* **Tác giả:** Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie
* **Hội nghị xuất bản:** **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)** (pp. 9268–9277).
* **DOI / Link:** [https://doi.org/10.1109/CVPR.2019.00949](https://doi.org/10.1109/CVPR.2019.00949)
* **Mục đích trích dẫn trong đồ án (Phục vụ RQ3):**
  * Cơ sở khoa học để thiết kế **Weighted Cross-Entropy Loss**.
  * Giải thích cơ chế phạt nặng hơn khi đoán sai các lớp thiểu số (Negative 21.2% và Neutral 15.0%) nhằm tối ưu hóa chỉ số **Macro F1-score** thay vì bị thiên vị theo lớp áp đảo (Positive 63.8%).
* **Mã BibTeX:**
```bibtex
@inproceedings{cui2019class,
  title={{Class-balanced loss based on effective number of samples}},
  author={Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9268--9277},
  year={2019}
}
```

---

## 2. NGUỒN BỘ DỮ LIỆU & THƯ VIỆN KỸ THUẬT (DATASET & OPEN-SOURCE TOOLS)

### [5] Bộ dữ liệu đánh giá sản phẩm Shopee (Shopee Reviews Dataset)
* **Tác giả:** L3viEvil (Lê Phúc Vĩnh Linh - `linhlpv`)
* **Nền tảng:** Kaggle Open Datasets (2018).
* **Đường dẫn truy cập:** [https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst)
* **Mục đích trích dẫn:** Nguồn gốc bộ dữ liệu gốc 31.460 dòng phục vụ nghiên cứu phi thương mại.

### [6] Thư viện Tách từ Tiếng Việt (PyVi Tokenizer)
* **Tác giả:** Xuan-Nga Cao (trungtv)
* **Kho lưu trữ:** [https://github.com/trungtv/pyvi](https://github.com/trungtv/pyvi)
* **Mục đích trích dẫn:** Dùng trong pipeline `src/preprocess.py` để tách từ ghép tiếng Việt (Word Segmentation) hỗ trợ RQ1.

### [7] Hugging Face Transformers Toolkit
* **Tác giả:** Wolf et al. (2020)
* **Xuất bản:** *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
* **Mục đích trích dẫn:** Thư viện nạp và tinh chỉnh (Fine-tuning) mô hình `vinai/phobert-base`.

---

## 3. BẢNG TRA CỨU TRÍCH DẪN NHANH TRONG BÁO CÁO (CITATION CHEAT SHEET)

Khi nhóm viết báo cáo Final Report cuối kỳ, gặp đoạn nào thì mở bảng này ra để trích dẫn số:

| Nội dung nhóm viết trong Báo cáo | Nguồn cần trích dẫn |
| :--- | :--- |
| Định nghĩa bài toán phân loại cảm xúc 3 nhãn tiếng Việt | Trích dẫn **[2] UIT-VSFC (2018)** |
| Nguồn gốc bộ dữ liệu Shopee 26.500 câu | Trích dẫn **[5] Kaggle Shopee (2018)** |
| Công cụ tách từ tiếng Việt chuẩn hóa văn bản (RQ1) | Trích dẫn **[6] PyVi Tokenizer** |
| Biện luận chọn kiến trúc mạng Bi-LSTM đọc 2 chiều (RQ2) | Trích dẫn **[3] Huang et al. (2015)** |
| Biện luận chọn VinAI PhoBERT thay vì mBERT (RQ2) | Trích dẫn **[1] PhoBERT VinAI (2020)** |
| Biện luận áp dụng hàm mất mát Weighted Loss chống lệch nhãn (RQ3) | Trích dẫn **[4] Cui et al. (2019)** |
