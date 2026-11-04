"""Generate Minh's Vietnamese report and Canva talking points from real outputs.

Run after ``python -m src.preprocess`` with
``python scripts/report_preprocessing.py`` from the repository root.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / "data/processed/preprocessing_audit.json").read_text(encoding="utf-8"))
    coverage = audit["vocabulary_coverage"]
    total = audit["retained_rows"]
    rules = audit["teencode_dictionary_entries"]
    changed = audit["teencode_changed_rows"]
    changed_pct = changed / audit["input_rows"]
    lines = [
        "# Báo cáo tiền xử lý dữ liệu Tuần 3",
        "",
        "Người thực hiện: Đặng Đức Minh (24020001), Data Engineer.",
        "",
        f"Đã hoàn thiện `src/preprocess.py`, từ điển {rules} quy tắc teencode, "
        "pipeline tách từ và bộ train/validation/test phục vụ các mô hình của nhóm. "
        "Các số liệu dưới đây được tính từ dữ liệu đang có trong repo, không phải kết quả mô phỏng.",
        "",
        "## Cách chạy và sử dụng",
        "",
        "Chạy các lệnh sau tại thư mục gốc của repo. Có thể dùng môi trường của nhóm với "
        "`requirements.txt`; môi trường tối thiểu riêng cho phần tiền xử lý được ghi trong "
        "`requirements-preprocess.txt`.",
        "",
        "```powershell",
        "python -m venv .venv",
        r".\.venv\Scripts\Activate.ps1",
        "python -m pip install -r requirements-preprocess.txt",
        "python -m unittest discover -s tests -v",
        "python -m src.preprocess",
        "python scripts/report_preprocessing.py",
        "```",
        "",
        "Tích hợp vào notebook hoặc module khác:",
        "",
        "```python",
        "from src.preprocess import preprocess_pipeline, split_and_save_stratified",
        'processed = preprocess_pipeline("SP ko tốt, ship chậm!")',
        'train, val, test = split_and_save_stratified(',
        '    "data/processed/shopee_reviews_cleaned.csv", "data/processed"',
        ")",
        "```",
        "",
        "## Pipeline và chính sách xử lý",
        "",
        "1. `clean_basic_text`: chuẩn hóa Unicode NFC, chữ thường, bỏ URL/email/số điện thoại "
        "di động Việt Nam, thẻ HTML, ký tự điều khiển và khoảng trắng thừa. Rút gọn chuỗi kéo dài "
        "từ 3 chữ cái cùng gốc trở lên, gồm `đẹpppppp → đẹp`, `quáaaaa → quá`; giữ chữ kép thông thường.",
        "2. `normalize_teencode`: tra từ điển cố định theo nguyên từ hoặc cụm từ, ưu tiên cụm dài. "
        "Ví dụ `ko → không`, `sp → sản phẩm`, `đc → được`, `shop → cửa hàng`, `ship → giao hàng`, "
        "`rep → trả lời`, `fb → phản hồi`. Không thay chữ `k` trong `100k` hoặc trong từ dài hơn.",
        "3. `segment_vietnamese`: dùng PyVi `ViTokenizer.tokenize`, sau đó nối 12 cụm từ TMĐT "
        "cố định nếu PyVi còn để rời, như `giao hàng → giao_hàng`. Đây là quy tắc bổ sung được công khai, "
        "không phải bảo đảm mọi từ ghép đều được nhận diện đúng.",
        "",
        "Giữ nguyên `content`, `label`, `start` của từng dòng đầu vào. Tạo `processed_content` "
        "và `label_id` theo `NEG=0`, `NEU=1`, `POS=2`. Không dùng số sao làm đặc trưng đầu vào. "
        "Không bỏ từ dừng mặc định để giữ từ phủ định, mức độ cảm xúc, dấu câu và emoji. "
        "`STOPWORDS_SET` được khai báo theo đặc tả và không chứa `không/chẳng/chưa`.",
        "",
        "## Kiểm tra dữ liệu và kết quả chia tập",
        "",
        f"File `shopee_reviews_cleaned.csv` thực tế có **{audit['input_rows']:,} dòng**, "
        "khác **26.458 dòng** trong README/kịch bản. Pipeline lấy số dòng từ file, không ép về con số trong kịch bản.",
        "",
        f"Có {audit['excluded_rows']:,} dòng rỗng hoặc không còn token từ/số sau xử lý; "
        f"các dòng này được lưu riêng vào `excluded_rows.csv` để kiểm tra. Giữ lại **{total:,} dòng**.",
        "",
        "| Tập | Số dòng | Tỷ lệ thực tế | NEG | NEU | POS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in audit["splits"].items():
        counts = item["label_counts"]
        lines.append(f"| {name} | {item['rows']:,} | {item['ratio']:.2%} | "
                     f"{counts['NEG']:,} | {counts['NEU']:,} | {counts['POS']:,} |")
    lines += [
        "",
        f"Seed cố định: **{audit['random_state']}**. Có **{audit['duplicate_groups']:,} nhóm** "
        "văn bản giống nhau sau tiền xử lý, "
        f"gồm {audit['extra_rows_in_duplicate_groups']:,} dòng bổ sung so với một đại diện mỗi nhóm. "
        f"Nhóm lớn nhất có {audit['largest_group_rows']:,} dòng.",
        "",
        "Để tránh cùng review xuất hiện trong nhiều tập, chia bảng nhóm bằng "
        "`train_test_split(..., stratify=...)` hai lần theo nhãn phổ biến nhất của nhóm, "
        "sau đó lấy lại toàn bộ các dòng trong mỗi nhóm. Trường hợp hòa chọn NEG, NEU, POS theo thứ tự ID. "
        "Nhãn của từng dòng vẫn giữ nguyên; tỷ lệ số dòng và phân bố nhãn vì vậy chỉ xấp xỉ mục tiêu 70/15/15. "
        "Đây là điều chỉnh có chủ đích so với chia ngẫu nhiên từng dòng trong đặc tả.",
        "",
        f"Số nội dung `processed_content` trùng giữa train–val / train–test / val–test: "
        f"**{audit['processed_overlap_between_splits']['train_val']} / "
        f"{audit['processed_overlap_between_splits']['train_test']} / "
        f"{audit['processed_overlap_between_splits']['val_test']}**.",
        "",
        f"Có **{audit['conflicting_label_groups']:,} nhóm** "
        f"({audit['conflicting_group_rows']:,} dòng) có cùng nội dung sau xử lý nhưng khác nhãn. "
        "Giữ mỗi nhóm trong cùng một tập; không tự sửa nhãn. Nhóm nên kiểm tra lại nhãn trước "
        "khi kết luận về chất lượng mô hình. Kiểm tra này không bảo đảm loại hết review gần trùng, "
        "trùng người dùng hoặc trùng sản phẩm vì chưa có các định danh đó.",
        "",
        "## Độ bao phủ từ điển và từ vựng",
        "",
        f"Từ điển có **{rules} mục**, **{audit['matched_effective_rules']} quy tắc** thực sự thay đổi "
        f"nội dung trong dữ liệu, với **{audit['replacement_occurrences']:,} lượt thay thế**. "
        f"Có **{changed:,}/{audit['input_rows']:,} dòng ({changed_pct:.2%})** thay đổi ở bước teencode. "
        "Tỷ lệ dòng thay đổi đo phạm vi tác động của từ điển, không đo độ chính xác chuẩn hóa.",
        "",
        "| Quy tắc có lượt thay thế cao | Số lượt |",
        "|---|---:|",
    ]
    for key, count in list(audit["replacement_counts"].items())[:12]:
        lines.append(f"| `{key}` | {count:,} |")
    lines += [
        "",
        f"Từ vựng tham chiếu gồm **{coverage['vocabulary_size']:,} token** xuất hiện ít nhất "
        f"2 lần trong train, trên tổng **{coverage['train_unique_tokens']:,} token khác nhau** ở train. "
        "Chỉ dùng tập train để tạo từ vựng; val/test chỉ được thống kê độ bao phủ. "
        "Token là từ/số sau tách từ, bỏ dấu câu và emoji khỏi phép đếm. Từ ghép có dấu gạch dưới "
        "được tính là một token. Đây không phải bộ đặc trưng TF-IDF 5.000 chiều của baseline.",
        "",
        "Độ bao phủ token = số lần xuất hiện của token thuộc từ vựng train / tổng số token. "
        "Độ bao phủ type = số token khác nhau thuộc từ vựng train / tổng số token khác nhau. "
        "OOV là token ngoài từ vựng tham chiếu.",
        "",
        "| Tập | Tổng token | Bao phủ token | Bao phủ type | Lượt token OOV |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in coverage["splits"].items():
        lines.append(f"| {name} | {item['tokens']:,} | {item['token_coverage']:.2%} | "
                     f"{item['type_coverage']:.2%} | {item['oov_tokens']:,} |")
    lines += [
        "",
        "Các ánh xạ `dc/dk`, `rep/fb`, từ không dấu và từ viết tắt ngắn có thể mơ hồ theo ngữ cảnh. "
        "Ở đây tuân theo quy ước của nhóm, không khẳng định mọi lần thay đều đúng. "
        "Không mở rộng từ điển theo lỗi trên test; mọi thay đổi tiếp theo cần dựa vào train "
        "và ghi nhận phiên bản trước khi đánh giá cuối cùng. Chưa đo mức cải thiện Macro F1; "
        "phần đó cần thí nghiệm ablation của Đình Hiếu.",
        "",
        "## Nội dung đưa vào Canva",
        "",
        "### Slide 2 Data Preprocessing Pipeline",
        "",
        "- Unicode NFC, lowercase; remove URLs, emails, phone numbers and redundant whitespace.",
        f"- Normalize Vietnamese e-commerce slang using {rules} fixed dictionary entries.",
        "- Preserve negation, punctuation and emojis.",
        "- Segment Vietnamese words with PyVi and a fixed commerce compound glossary.",
        f"- Teencode normalization changes {changed:,} reviews ({changed_pct:.2%}); "
        "classification impact will be measured by ablation.",
        "- Example: `SP ko tốt, ship chậm! → sản_phẩm không tốt , giao_hàng chậm !`",
        "",
        "Lời trình bày: Dạ thưa Thầy, em phụ trách pipeline tiền xử lý. Em chuẩn hóa Unicode, "
        "chữ thường, thông tin liên hệ và các ký tự kéo dài; sau đó thay teencode theo nguyên từ "
        "để không làm sai các trường hợp như giá 100k. Cuối cùng em tách từ bằng PyVi và bổ sung "
        "một số cụm từ thương mại điện tử cố định. Em giữ từ phủ định như không, chẳng, chưa "
        "vì các từ này có thể đảo ngược cảm xúc. Số liệu độ bao phủ cho thấy pipeline tác động "
        "đến bao nhiêu review; để biết có cải thiện mô hình hay không, nhóm sẽ so sánh bằng ablation.",
        "",
        "### Slide 3 Stratified Data Splitting",
        "",
        f"- Actual input: {audit['input_rows']:,} reviews; retained after preprocessing: {total:,}.",
        "- Target split: 70% train / 15% validation / 15% test; seed 42.",
        "- Keep identical processed reviews in the same split; stratify groups by their modal label.",
        "- Ratios are approximate because duplicate groups remain intact.",
        "- No exact processed review overlaps across splits. Fit vocabularies on train only.",
        "",
        "Lời trình bày: Dữ liệu hiện có trong repo khác số dòng trong kịch bản cũ nên em báo cáo "
        "theo file thực tế. Sau chuẩn hóa, một số câu khác cách viết trở thành cùng nội dung. "
        "Em giữ chúng trong cùng một tập rồi chia phân tầng ở cấp nhóm, nhằm tránh cùng review "
        "xuất hiện ở cả tập học và tập đánh giá. Vì giữ nguyên nhóm nên tỷ lệ chỉ xấp xỉ 70/15/15. "
        "Các dòng khác nhãn vẫn giữ nguyên để nhóm kiểm tra, không tự động sửa nhãn.",
        "",
        "## Bàn giao và tái lập",
        "",
        "- `src/preprocess.py`: đầy đủ 5 hàm theo đặc tả, CLI và đo độ bao phủ.",
        "- `tests/test_preprocess.py`: kiểm thử Unicode, teencode, PyVi, nhãn, schema, "
        "tái lập, giữ nguyên dữ liệu nguồn và kiểm soát trùng giữa các tập.",
        "- `data/processed/train.csv`, `val.csv`, `test.csv`: cùng 5 cột chuẩn "
        "`content, processed_content, label, label_id, start`.",
        "- `data/processed/excluded_rows.csv`: các dòng cần kiểm tra vì không còn từ/số.",
        "- `data/processed/preprocessing_audit.json`: số liệu chi tiết, tần suất quy tắc, "
        "tỷ lệ nhãn, OOV và phiên bản thư viện.",
        "",
        f"SHA-256 của dữ liệu đầu vào: `{audit['source_sha256']}`. "
        f"Kiểm tra dữ liệu nguồn không đổi: **{audit['source_unchanged']}**.",
        "",
        "Phiên bản đã chạy: " + ", ".join(f"{name} {value}" for name, value in audit["versions"].items()) + ".",
        "",
        "Tài liệu thư viện: [PyVi](https://github.com/trungtv/pyvi), "
        "[scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).",
        "",
    ]
    target = ROOT / "docs/MINH_WEEK3_PREPROCESSING_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
