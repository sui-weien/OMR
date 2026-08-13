import unittest

import cv2
import numpy as np

from primitive_omr.detector import (
    DetectionConfig,
    PrimitiveDetector,
    _classify_accidental_shape,
    _merge_vertical_lines,
    _thick_bridge_coverage,
    detect_accidentals,
    estimate_staff_spacing,
)


class PrimitiveDetectorTest(unittest.TestCase):
    def make_score(self):
        image = np.full((300, 600, 3), 255, dtype=np.uint8)
        note_score = np.zeros((300, 600), dtype=np.float32)
        stem_score = np.zeros((300, 600), dtype=np.float32)
        for y in (120, 132, 144, 156, 168):
            cv2.line(image, (40, y), (560, y), (0, 0, 0), 1)

        for center_x in (200, 300, 430):
            cv2.ellipse(image, (center_x, 150), (8, 6), 0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(note_score, (center_x, 150), (8, 6), 0, 0, 360, 0.95, -1)
            cv2.line(image, (center_x + 8, 96), (center_x + 8, 150), (0, 0, 0), 2)
            cv2.line(stem_score, (center_x + 8, 96), (center_x + 8, 150), 0.9, 2)
        beam = np.asarray([[208, 92], [308, 97], [308, 104], [208, 99]], dtype=np.int32)
        cv2.fillConvexPoly(image, beam, (0, 0, 0))
        flag = np.asarray([[438, 96], [450, 104], [455, 118], [448, 128]], dtype=np.int32)
        cv2.polylines(image, [flag], False, (0, 0, 0), 4, cv2.LINE_AA)
        return image, note_score, stem_score

    def test_staff_spacing(self):
        image, _, _ = self.make_score()
        spacing = estimate_staff_spacing(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), DetectionConfig()
        )
        self.assertAlmostEqual(spacing, 12.0, delta=1.0)

    def test_merges_multiple_hough_edges_of_one_stem(self):
        merged = _merge_vertical_lines(
            [(100, 20, 100, 80), (104, 22, 104, 78), (107, 21, 107, 79), (125, 20, 125, 80)],
            spacing=12.0,
        )
        self.assertEqual(len(merged), 2)

    def test_beam_bridge_rejects_thin_staff_line(self):
        staff = np.zeros((100, 160), dtype=np.uint8)
        beam = np.zeros_like(staff)
        cv2.line(staff, (20, 50), (140, 50), 255, 1)
        cv2.rectangle(beam, (20, 46), (140, 54), 255, -1)
        staff_coverage = _thick_bridge_coverage(staff, (20, 50), (140, 50), 12.0)
        beam_coverage = _thick_bridge_coverage(beam, (20, 50), (140, 50), 12.0)
        self.assertLess(staff_coverage, 0.1)
        self.assertGreater(beam_coverage, 0.9)

    def test_detects_and_links_primitives(self):
        image, note_score, stem_score = self.make_score()
        result = PrimitiveDetector().detect(image, note_score, stem_score, source="synthetic")
        self.assertGreaterEqual(result["counts"]["noteheads"], 2)
        self.assertGreaterEqual(result["counts"]["stems"], 2)
        self.assertGreaterEqual(result["counts"]["beams"], 1)
        self.assertGreaterEqual(result["counts"]["flags"], 1)
        self.assertGreaterEqual(len(result["relations"]["notehead_to_stem"]), 2)
        self.assertGreaterEqual(len(result["relations"]["stem_to_beam"]), 2)

    def test_direct_flag_segmentation_channel(self):
        image, note_score, stem_score = self.make_score()
        flag_score = np.zeros(image.shape[:2], dtype=np.float32)
        flag = np.asarray([[438, 96], [450, 104], [455, 118], [448, 128]], dtype=np.int32)
        cv2.polylines(flag_score, [flag], False, 0.98, 4, cv2.LINE_AA)
        result = PrimitiveDetector().detect(
            image,
            note_score,
            stem_score,
            flag_score=flag_score,
            source="synthetic-direct-flag",
        )
        direct = [
            item
            for item in result["flags"]
            if item.get("source") == "direct-flag-segmentation"
        ]
        self.assertGreaterEqual(len(direct), 1)
        self.assertGreaterEqual(len(result["relations"]["stem_to_flag"]), 1)

    def test_detect_accidental_links_sharp_to_notehead(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        spacing = 12.0
        noteheads = [{"id": "notehead-0", "bbox": [90, 94, 106, 106], "center": [98.0, 100.0]}]
        # Two vertical strokes crossed by two horizontal bars that overhang
        # past the verticals on both sides, the way a printed sharp looks.
        cv2.line(gray, (76, 90), (76, 112), 0, 2)
        cv2.line(gray, (82, 90), (82, 112), 0, 2)
        cv2.line(gray, (73, 97), (85, 97), 0, 2)
        cv2.line(gray, (73, 105), (85, 105), 0, 2)
        accidentals, links = detect_accidentals(gray, noteheads, [], spacing)
        self.assertEqual(len(accidentals), 1)
        self.assertEqual(links, [{"notehead_id": "notehead-0", "accidental_id": accidentals[0]["id"]}])
        self.assertEqual(accidentals[0]["type"], "sharp")

    def test_classify_accidental_shape_detects_flat_via_enclosed_hole(self):
        glyph = np.zeros((26, 16), dtype=np.uint8)
        cv2.line(glyph, (3, 2), (3, 24), 255, 2)
        cv2.circle(glyph, (9, 18), 5, 255, 2)
        accidental_type, confidence = _classify_accidental_shape(glyph)
        self.assertEqual(accidental_type, "flat")
        self.assertGreater(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
