from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from primitive_omr.musicxml import (
    build_note_events,
    detect_staffs,
    staff_y_to_pitch,
    write_musicxml,
)


class MusicXMLConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((260, 500, 3), 255, dtype=np.uint8)
        for top in (50, 150):
            for line in range(5):
                cv2.line(self.image, (20, top + 10 * line), (480, top + 10 * line), (0, 0, 0), 1)

    def test_detect_two_piano_staffs_and_pitch(self):
        staffs = detect_staffs(self.image, 10.0, "piano")
        self.assertEqual(len(staffs), 2)
        self.assertEqual(staffs[0].clef, "treble")
        self.assertEqual(staffs[1].clef, "bass")
        self.assertEqual(staff_y_to_pitch(90, staffs[0]), "E4")
        self.assertEqual(staff_y_to_pitch(190, staffs[1]), "G2")

    def test_single_bass_staff_mode(self):
        staffs = detect_staffs(self.image[:130], 10.0, "bass")
        self.assertEqual(len(staffs), 1)
        self.assertEqual(staffs[0].clef, "bass")
        self.assertEqual(staff_y_to_pitch(90, staffs[0]), "G2")

    def test_build_event_and_write_parseable_xml(self):
        cv2.ellipse(self.image, (150, 85), (7, 5), 0, 0, 360, (0, 0, 0), -1)
        detection = {
            "source": "/tmp/test.png",
            "staff_spacing": 10.0,
            "noteheads": [{"id": "notehead-0", "bbox": [143, 80, 157, 90], "center": [150, 85]}],
            "stems": [{"id": "stem-0", "line": [157, 50, 157, 85]}],
            "beams": [{"id": "beam-0", "bbox": [157, 49, 210, 54]}],
            "flags": [],
            "relations": {
                "notehead_to_stem": [{"notehead_id": "notehead-0", "stem_id": "stem-0"}],
                "stem_to_beam": [{"stem_id": "stem-0", "beam_id": "beam-0"}],
                "stem_to_flag": [],
            },
        }
        staffs = detect_staffs(self.image, 10.0, "piano")
        events, warnings = build_note_events(self.image, detection, staffs)
        self.assertFalse(warnings)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].quarter_length, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "score.musicxml"
            write_musicxml(events, staffs, output, "test")
            self.assertTrue(output.exists())
            from music21 import converter

            parsed = converter.parse(str(output))
            self.assertGreater(len(parsed.parts), 0)


if __name__ == "__main__":
    unittest.main()
