"""Behavior and data-integrity tests; run: python -m unittest discover -s tests -v."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocess import (
    LABEL_TO_ID, NEGATIONS, OUTPUT_COLUMNS, STOPWORDS_SET, TEENCODE_DICT,
    clean_basic_text, normalize_teencode, preprocess_pipeline,
    segment_vietnamese, split_and_save_stratified, vocabulary_coverage,
)


class TextTests(unittest.TestCase):
    def test_unicode_whitespace_and_elongation(self):
        self.assertEqual(clean_basic_text("  A\u0301O\tđẹpppppp quáaaaa\n "), "áo đẹp quá")
        self.assertEqual(clean_basic_text("good cool"), "good cool")

    def test_contacts_removed_but_prices_and_sentiment_retained(self):
        text = "SP ko tốt ❤️! 100k 5 sao https://x.vn/a www.x.vn a@x.vn 0901.234.567 +84 901 234 567"
        self.assertEqual(clean_basic_text(text), "sp ko tốt ❤️! 100k 5 sao")

    def test_missing_values_and_html(self):
        for value in (None, float("nan"), pd.NA):
            self.assertEqual(clean_basic_text(value), "")
        self.assertEqual(clean_basic_text("<b>ÁO</b>&nbsp;đẹp\u200b!"), "áo đẹp !")

    def test_dictionary_and_required_aliases(self):
        self.assertGreater(sum(k != v for k, v in TEENCODE_DICT.items()), 100)
        self.assertTrue(STOPWORDS_SET.isdisjoint(NEGATIONS))
        self.assertEqual(normalize_teencode("k ko khong sp đc dk dc shop ship auth aut rep fb", TEENCODE_DICT),
                         "không không không sản phẩm được được được cửa hàng giao hàng chính hãng chính hãng trả lời phản hồi")

    def test_whole_words_phrases_and_custom_dictionary(self):
        self.assertEqual(normalize_teencode("SP k tốt, giá 100k skincare size_k!", TEENCODE_DICT),
                         "sản phẩm không tốt, giá 100k skincare size_k!")
        self.assertEqual(normalize_teencode("khối lượng 1 kg giá 100k", TEENCODE_DICT),
                         "khối lượng 1 kg giá 100k")
        self.assertEqual(normalize_teencode("free ship và ship", TEENCODE_DICT),
                         "miễn phí giao hàng và giao hàng")
        self.assertEqual(normalize_teencode("a b", {"a": "b", "b": "c"}), "b c")
        self.assertEqual(normalize_teencode("giữ nguyên", {}), "giữ nguyên")
        with self.assertRaises(ValueError):
            normalize_teencode("abc", {"": "x"})

    def test_real_pyvi_segmentation_and_pipeline(self):
        self.assertEqual(segment_vietnamese("sản phẩm rất tốt giao hàng nhanh"),
                         "sản_phẩm rất tốt giao_hàng nhanh")
        output = preprocess_pipeline("SP này ko tốt, chất lượng kém!")
        self.assertIn("sản_phẩm", output)
        self.assertIn("không", output)
        self.assertIn("chất_lượng", output)
        self.assertEqual(segment_vietnamese(""), "")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input.csv"
        rows = [{"content": f"sản phẩm mã {index} {label}", "label": label, "start": stars}
                for label, stars in (("NEG", 1), ("NEU", 3), ("POS", 5)) for index in range(40)]
        # Both same-label duplicates and conflicting supplied labels must stay
        # together. The pipeline must not change a supplied sentiment label.
        rows.extend([{"content": "sp ko tốt", "label": "NEG", "start": 1},
                     {"content": "sản phẩm không tốt", "label": "NEG", "start": 2},
                     {"content": "SP KO TỐT", "label": "POS", "start": 5},
                     {"content": "https://example.vn", "label": "NEU", "start": 3}])
        self.original = pd.DataFrame(rows)
        self.original.to_csv(self.source, index=False, encoding="utf-8-sig")

    def tearDown(self):
        self.temp.cleanup()

    def test_split_preserves_rows_labels_schema_and_source(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        parts = split_and_save_stratified(str(self.source), str(self.root / "out"))
        combined = pd.concat(parts, ignore_index=True)
        expected = self.original.iloc[:-1].copy()
        self.assertEqual(len(combined), len(expected))
        for part in parts:
            self.assertEqual(list(part.columns), OUTPUT_COLUMNS)
            self.assertEqual(set(part.label), set(LABEL_TO_ID))
            self.assertTrue(part.label_id.eq(part.label.map(LABEL_TO_ID)).all())
            self.assertTrue(part.processed_content.str.len().gt(0).all())
        actual_records = sorted(combined[["content", "label", "start"]].itertuples(index=False, name=None))
        self.assertEqual(actual_records, sorted(expected.itertuples(index=False, name=None)))
        key_sets = [set(part.processed_content) for part in parts]
        self.assertFalse(key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2])
        duplicate_key = preprocess_pipeline("sp ko tốt")
        self.assertEqual(sum(part.processed_content.eq(duplicate_key).any() for part in parts), 1)
        audit = json.loads((self.root / "out/preprocessing_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["excluded_rows"], 1)
        self.assertEqual(audit["conflicting_label_groups"], 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
        for name, part in zip(("train", "val", "test"), parts):
            loaded = pd.read_csv(self.root / f"out/{name}.csv", keep_default_na=False)
            pd.testing.assert_frame_equal(loaded, part)

    def test_seed_reproducibility_and_proportions(self):
        first = split_and_save_stratified(str(self.source), str(self.root / "one"))
        second = split_and_save_stratified(str(self.source), str(self.root / "two"))
        for one, two, target in zip(first, second, (.70, .15, .15)):
            pd.testing.assert_frame_equal(one, two)
            self.assertLess(abs(len(one) / 123 - target), .03)
            for label in LABEL_TO_ID:
                self.assertLess(abs(one.label.eq(label).mean() - 1 / 3), .07)

    def test_bad_ratios_labels_schema_and_overwrite_are_rejected(self):
        for ratio in ((.8, .15, .15), (.7, .3, 0), (float("nan"), .15, .15)):
            with self.assertRaises(ValueError):
                split_and_save_stratified(str(self.source), str(self.root / "bad"), *ratio)
        unknown = self.original.copy()
        unknown.loc[0, "label"] = "UNKNOWN"
        unknown.to_csv(self.root / "unknown.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Unknown labels"):
            split_and_save_stratified(str(self.root / "unknown.csv"), str(self.root / "bad"))
        self.original.drop(columns="start").to_csv(self.root / "missing.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            split_and_save_stratified(str(self.root / "missing.csv"), str(self.root / "bad"))
        self.original.to_csv(self.root / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "overwritten"):
            split_and_save_stratified(str(self.root / "train.csv"), str(self.root))

    def test_coverage_vocabulary_fits_train_only(self):
        train = pd.DataFrame({"processed_content": ["áo đẹp đẹp", "áo tốt"]})
        heldout = pd.DataFrame({"processed_content": ["áo từ_mới"]})
        result = vocabulary_coverage(train, heldout, heldout)
        self.assertEqual(result["vocabulary_size"], 2)
        self.assertEqual(result["splits"]["val"]["token_coverage"], .5)
        self.assertEqual(result["splits"]["val"]["oov_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
