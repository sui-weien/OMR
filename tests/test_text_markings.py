import unittest

from primitive_omr.text_markings import classify_text_marking


class TextMarkingClassificationTest(unittest.TestCase):
    def test_recognizes_dynamic_marks(self):
        for text in ("p", "f", "pp", "ff", "sf", "sfz", "cresc.", "decresc."):
            self.assertEqual(classify_text_marking(text, 0.5), "dynamic_mark")

    def test_recognizes_expression_marks(self):
        for text in ("dolce", "sempre", "poco", "espressivo", "Voce"):
            self.assertEqual(classify_text_marking(text, 0.5), "expression_mark")

    def test_ignores_header_region_regardless_of_text(self):
        self.assertIsNone(classify_text_marking("p", 0.02))
        self.assertIsNone(classify_text_marking("dolce", 0.0))

    def test_ignores_unrecognized_and_garbled_text(self):
        # These are real OCR misreads of a stylized "Ped." glyph observed
        # while validating against ground truth; they must not be
        # misclassified just because they superficially resemble a mark.
        for text in ("Tea.", "Teo.", "Tel.", "general word", "Sonata", "3"):
            self.assertIsNone(classify_text_marking(text, 0.5))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(classify_text_marking("  Cresc.  ", 0.5), "dynamic_mark")
        self.assertEqual(classify_text_marking("DOLCE", 0.5), "expression_mark")


if __name__ == "__main__":
    unittest.main()
