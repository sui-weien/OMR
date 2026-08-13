"""Turn primitive detections into a conservative, parseable MusicXML score.

This module deliberately keeps the conversion assumptions explicit.  The
primitive detector currently recognises noteheads, stems, beams, flags, and
note-level sharp/flat/natural accidentals; it does not yet recognise clefs,
key signatures, rests or time signatures.  Staff geometry and pitch are
therefore inferred from the page image and any accidental printed directly
next to a note, while missing musical symbols are reported in the conversion
summary.  Accidental type classification is an unverified geometry heuristic
(no trained model or labelled dataset backs it) and accidentals are never
carried forward to later notes in the same measure, since barlines are not
yet detected reliably enough to bound a measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class StaffGeometry:
    index: int
    lines: tuple[float, float, float, float, float]
    spacing: float
    clef: str
    track: int
    system: int

    @property
    def center(self) -> float:
        return float(np.mean(self.lines))


@dataclass
class NoteEvent:
    staff_index: int
    x: float
    notehead_ids: list[str]
    stem_id: str | None
    pitches: list[str]
    quarter_length: float
    rhythm_source: str
    beam_levels: int
    flag_levels: int


def _row_peaks(projection: np.ndarray, minimum: float) -> list[int]:
    """Return local maxima without merging nearby notation into one wide run."""
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            projection,
            height=minimum,
            distance=3,
            prominence=max(4.0, minimum * 0.12),
        )
        return [int(row) for row in peaks]
    except ImportError:  # pragma: no cover - scipy is part of the project env
        pass

    # Dependency-light fallback.  It is intentionally a local-max test rather
    # than connected-run grouping: dense chords and beams can connect several
    # actual staff rows into one large foreground run.
    rows = np.flatnonzero(projection >= minimum)
    if rows.size == 0:
        return []
    peaks: list[int] = []
    for row in rows:
        y = int(row)
        left = max(0, y - 2)
        right = min(len(projection), y + 3)
        if projection[y] >= float(np.max(projection[left:right])):
            if not peaks or y - peaks[-1] >= 3:
                peaks.append(y)
            elif projection[y] > projection[peaks[-1]]:
                peaks[-1] = y
    return peaks


def detect_staffs(
    image: np.ndarray,
    staff_spacing: float,
    staff_mode: str = "piano",
) -> list[StaffGeometry]:
    """Detect five-line staffs from long horizontal strokes in a page image."""
    if staff_mode not in {"piano", "treble", "bass"}:
        raise ValueError("staff_mode must be piano, treble, or bass")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    kernel_width = max(25, int(round(width * 0.08)))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    left = int(round(width * 0.10))
    right = max(left + 1, int(round(width * 0.96)))
    projection = np.count_nonzero(horizontal[:, left:right], axis=1).astype(np.float32)
    peaks = _row_peaks(projection, minimum=max(20.0, (right - left) * 0.10))
    if len(peaks) < 5:
        raise ValueError("Could not find five staff lines in the image")

    spacing = float(staff_spacing)
    tolerance = max(2.5, spacing * 0.32)
    candidates: dict[tuple[int, ...], float] = {}
    for top in peaks:
        matched: list[int] = []
        for line_index in range(5):
            expected = top + line_index * spacing
            nearest = min(peaks, key=lambda y: abs(y - expected))
            if abs(nearest - expected) > tolerance:
                break
            matched.append(nearest)
        if len(matched) != 5 or len(set(matched)) != 5:
            continue
        gaps = np.diff(matched).astype(np.float32)
        local_spacing = float(np.median(gaps))
        if not 0.72 * spacing <= local_spacing <= 1.28 * spacing:
            continue
        if float(np.max(np.abs(gaps - local_spacing))) > tolerance:
            continue
        key = tuple(matched)
        regularity_penalty = 25.0 * float(np.sum(np.abs(gaps - spacing)))
        score = float(sum(projection[y] for y in matched)) - regularity_penalty
        candidates[key] = max(candidates.get(key, -np.inf), score)

    # Notes and beams can form five parallel horizontal rows.  They may look
    # like another staff displaced only a few pixels from the real one.  Keep
    # only the strongest candidate within one staff-height neighbourhood.
    selected: list[tuple[int, ...]] = []
    for lines, _score in sorted(candidates.items(), key=lambda item: item[1], reverse=True):
        center = float(np.mean(lines))
        if any(abs(center - float(np.mean(other))) < spacing * 4.2 for other in selected):
            continue
        selected.append(lines)
    selected.sort(key=lambda lines: float(np.mean(lines)))
    if not selected:
        raise ValueError("Staff-line candidates were found, but none formed a valid staff")

    staffs: list[StaffGeometry] = []
    for index, lines in enumerate(selected):
        if staff_mode == "piano":
            clef_name = "treble" if index % 2 == 0 else "bass"
            track = index % 2
            system = index // 2
        else:
            clef_name = staff_mode
            track = 0
            system = index
        staffs.append(
            StaffGeometry(
                index=index,
                lines=tuple(float(value) for value in lines),  # type: ignore[arg-type]
                spacing=float(np.median(np.diff(lines))),
                clef=clef_name,
                track=track,
                system=system,
            )
        )
    return staffs


_DIATONIC_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_ACCIDENTAL_SYMBOLS = {"sharp": "#", "flat": "-"}
_ACCIDENTAL_CONFIDENCE_THRESHOLD = 0.5


def staff_y_to_pitch(y: float, staff: StaffGeometry) -> str:
    """Convert a y coordinate to the nearest natural diatonic staff position."""
    bottom_line_pitch = ("E", 4) if staff.clef == "treble" else ("G", 2)
    bottom_index = _DIATONIC_STEPS.index(bottom_line_pitch[0]) + 7 * bottom_line_pitch[1]
    half_space = max(1.0, staff.spacing / 2.0)
    staff_steps = int(round((staff.lines[-1] - y) / half_space))
    absolute_index = bottom_index + staff_steps
    octave, step_index = divmod(absolute_index, 7)
    return f"{_DIATONIC_STEPS[step_index]}{octave}"


def _notehead_is_filled(image: np.ndarray, notehead: dict[str, Any]) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    x0, y0, x1, y1 = (int(value) for value in notehead["bbox"])
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    # Inspect the centre: hollow noteheads retain a white core, whereas filled
    # noteheads remain dark.  Cropping the outer 20% avoids most attached stems.
    xa = max(0, x0 + int(round(width * 0.20)))
    xb = min(gray.shape[1], x1 - int(round(width * 0.20)) + 1)
    ya = max(0, y0 + int(round(height * 0.20)))
    yb = min(gray.shape[0], y1 - int(round(height * 0.20)) + 1)
    crop = gray[ya:yb, xa:xb]
    if crop.size == 0:
        return True
    return float(np.mean(crop < 170)) >= 0.48


def _beam_level_count(
    stem_id: str,
    beams_by_id: dict[str, dict[str, Any]],
    beam_links: dict[str, list[str]],
    spacing: float,
) -> int:
    beam_ids = beam_links.get(stem_id, [])
    centers = sorted(
        float(np.mean(beams_by_id[beam_id]["bbox"][1::2]))
        for beam_id in beam_ids
        if beam_id in beams_by_id
    )
    distinct: list[float] = []
    for center in centers:
        if not distinct or abs(center - distinct[-1]) > max(2.0, spacing * 0.28):
            distinct.append(center)
    return min(4, len(distinct))


def _nearest_staff(y: float, staffs: list[StaffGeometry]) -> StaffGeometry:
    def distance(staff: StaffGeometry) -> float:
        if staff.lines[0] <= y <= staff.lines[-1]:
            return 0.0
        return min(abs(y - staff.lines[0]), abs(y - staff.lines[-1]))

    return min(staffs, key=distance)


def build_note_events(
    image: np.ndarray,
    detection: dict[str, Any],
    staffs: list[StaffGeometry],
) -> tuple[list[NoteEvent], list[str]]:
    """Resolve primitive links into pitched, ordered note/chord events."""
    notes_by_id = {note["id"]: note for note in detection.get("noteheads", [])}
    stems_by_id = {stem["id"]: stem for stem in detection.get("stems", [])}
    beams_by_id = {beam["id"]: beam for beam in detection.get("beams", [])}
    flags_by_id = {flag["id"]: flag for flag in detection.get("flags", [])}
    accidentals_by_id = {acc["id"]: acc for acc in detection.get("accidentals", [])}
    relations = detection.get("relations", {})

    accidental_by_notehead: dict[str, dict[str, Any]] = {}
    for relation in relations.get("notehead_to_accidental", []):
        accidental = accidentals_by_id.get(relation["accidental_id"])
        if accidental is not None:
            accidental_by_notehead[relation["notehead_id"]] = accidental

    def apply_accidental(pitch: str, note_id: str) -> str:
        accidental = accidental_by_notehead.get(note_id)
        if accidental is None:
            return pitch
        if accidental.get("type_confidence", 0.0) < _ACCIDENTAL_CONFIDENCE_THRESHOLD:
            return pitch
        symbol = _ACCIDENTAL_SYMBOLS.get(accidental.get("type"))
        if symbol is None:
            # "natural" cancels an assumed alteration, which this project does
            # not model (no key signature), and "unknown" is not applied.
            return pitch
        return pitch[0] + symbol + pitch[1:]

    candidate_stems: dict[str, list[str]] = {}
    for relation in relations.get("notehead_to_stem", []):
        if relation["notehead_id"] in notes_by_id and relation["stem_id"] in stems_by_id:
            candidate_stems.setdefault(relation["notehead_id"], []).append(relation["stem_id"])

    def stem_cost(note: dict[str, Any], stem: dict[str, Any]) -> float:
        x0, _y0, x1, _y1 = note["bbox"]
        nx, ny = note["center"]
        sx0, sy0, sx1, sy1 = stem["line"]
        sx = (sx0 + sx1) / 2.0
        horizontal = min(abs(sx - x0), abs(sx - x1), abs(sx - nx))
        endpoint = min(abs(sy0 - ny), abs(sy1 - ny))
        return horizontal + 0.22 * endpoint

    chosen_stem: dict[str, str] = {}
    for note_id, stem_ids in candidate_stems.items():
        chosen_stem[note_id] = min(
            stem_ids,
            key=lambda stem_id: stem_cost(notes_by_id[note_id], stems_by_id[stem_id]),
        )

    groups: dict[str, list[str]] = {}
    for note_id, stem_id in chosen_stem.items():
        groups.setdefault(f"stem:{stem_id}", []).append(note_id)
    unlinked = [note_id for note_id in notes_by_id if note_id not in chosen_stem]
    # Unlinked noteheads at effectively the same x coordinate are probably a
    # chord sharing a missed stem (or a stemless whole-note chord).
    for note_id in sorted(unlinked, key=lambda item: notes_by_id[item]["center"][0]):
        note = notes_by_id[note_id]
        key = None
        for existing_key, members in groups.items():
            if not existing_key.startswith("unlinked:"):
                continue
            mean_x = float(np.mean([notes_by_id[item]["center"][0] for item in members]))
            if abs(note["center"][0] - mean_x) <= detection["staff_spacing"] * 0.55:
                key = existing_key
                break
        if key is None:
            key = f"unlinked:{note_id}"
        groups.setdefault(key, []).append(note_id)

    beam_links: dict[str, list[str]] = {}
    for relation in relations.get("stem_to_beam", []):
        beam_links.setdefault(relation["stem_id"], []).append(relation["beam_id"])
    flag_links: dict[str, list[str]] = {}
    for relation in relations.get("stem_to_flag", []):
        flag_links.setdefault(relation["stem_id"], []).append(relation["flag_id"])

    events: list[NoteEvent] = []
    warnings: list[str] = []
    for group_key, note_ids in groups.items():
        noteheads = [notes_by_id[note_id] for note_id in note_ids]
        x = float(np.mean([note["center"][0] for note in noteheads]))
        y = float(np.median([note["center"][1] for note in noteheads]))
        staff = _nearest_staff(y, staffs)
        max_ledger_distance = staff.spacing * 4.5
        if y < staff.lines[0] - max_ledger_distance or y > staff.lines[-1] + max_ledger_distance:
            warnings.append(
                f"Skipped {group_key}: notehead group is too far from every detected staff"
            )
            continue

        stem_id = group_key.removeprefix("stem:") if group_key.startswith("stem:") else None
        beam_levels = (
            _beam_level_count(stem_id, beams_by_id, beam_links, staff.spacing)
            if stem_id
            else 0
        )
        flag_levels = 0
        if stem_id:
            flag_levels = max(
                (
                    int(flags_by_id[flag_id].get("count", 1))
                    for flag_id in flag_links.get(stem_id, [])
                    if flag_id in flags_by_id
                ),
                default=0,
            )
        rhythmic_levels = max(beam_levels, flag_levels)
        filled = all(_notehead_is_filled(image, note) for note in noteheads)
        if rhythmic_levels:
            quarter_length = 1.0 / (2**rhythmic_levels)
            rhythm_source = "beam" if beam_levels >= flag_levels else "flag"
        elif stem_id:
            quarter_length = 1.0 if filled else 2.0
            rhythm_source = "filled-notehead+stem" if filled else "hollow-notehead+stem"
        else:
            quarter_length = 1.0 if filled else 4.0
            rhythm_source = "unlinked-filled-notehead" if filled else "hollow-notehead"
        pitches = sorted(
            {
                apply_accidental(staff_y_to_pitch(float(note["center"][1]), staff), note["id"])
                for note in noteheads
            },
            key=lambda pitch: int(pitch[-1]) * 7 + _DIATONIC_STEPS.index(pitch[0]),
        )
        events.append(
            NoteEvent(
                staff_index=staff.index,
                x=x,
                notehead_ids=note_ids,
                stem_id=stem_id,
                pitches=pitches,
                quarter_length=quarter_length,
                rhythm_source=rhythm_source,
                beam_levels=beam_levels,
                flag_levels=flag_levels,
            )
        )
    events.sort(key=lambda event: (event.staff_index, event.x))
    return events, warnings


def write_musicxml(
    events: list[NoteEvent],
    staffs: list[StaffGeometry],
    output_path: Path,
    title: str,
    beats: int = 4,
    beat_type: int = 4,
) -> None:
    """Write events to a multi-part MusicXML score with music21."""
    try:
        from music21 import chord, clef, metadata, meter, note, stream
    except ImportError as exc:  # pragma: no cover - environment error path
        raise RuntimeError("music21 is required to export MusicXML") from exc

    score = stream.Score(id="PrimitiveOMRScore")
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "OMR transcription (verify before use)"
    track_count = max(staff.track for staff in staffs) + 1
    measure_capacity = beats * (4.0 / beat_type)
    for track in range(track_count):
        track_staffs = [staff for staff in staffs if staff.track == track]
        part = stream.Part(id=f"P{track + 1}")
        part.partName = "Right Hand" if track_count == 2 and track == 0 else (
            "Left Hand" if track_count == 2 else f"Staff {track + 1}"
        )
        measure_number = 1
        measure = stream.Measure(number=measure_number)
        measure.insert(0, meter.TimeSignature(f"{beats}/{beat_type}"))
        clef_name = track_staffs[0].clef
        measure.insert(0, clef.TrebleClef() if clef_name == "treble" else clef.BassClef())
        elapsed = 0.0
        track_events: list[NoteEvent] = []
        for staff in track_staffs:
            track_events.extend(event for event in events if event.staff_index == staff.index)
        track_events.sort(key=lambda event: (staffs[event.staff_index].system, event.x))
        previous_system = None
        for event in track_events:
            system = staffs[event.staff_index].system
            if previous_system is not None and system != previous_system and len(measure.notes) > 0:
                part.append(measure)
                measure_number += 1
                measure = stream.Measure(number=measure_number)
                elapsed = 0.0
            if elapsed > 0 and elapsed + event.quarter_length > measure_capacity + 1e-6:
                part.append(measure)
                measure_number += 1
                measure = stream.Measure(number=measure_number)
                elapsed = 0.0
            element = (
                chord.Chord(event.pitches, quarterLength=event.quarter_length)
                if len(event.pitches) > 1
                else note.Note(event.pitches[0], quarterLength=event.quarter_length)
            )
            element.lyric = None
            measure.append(element)
            elapsed += event.quarter_length
            previous_system = system
            if elapsed >= measure_capacity - 1e-6:
                part.append(measure)
                measure_number += 1
                measure = stream.Measure(number=measure_number)
                elapsed = 0.0
        if len(measure.notes) > 0 or len(part.getElementsByClass(stream.Measure)) == 0:
            part.append(measure)
        score.append(part)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(output_path))


def draw_musicxml_overlay(
    image: np.ndarray,
    primitive_overlay: np.ndarray,
    staffs: list[StaffGeometry],
    events: list[NoteEvent],
) -> np.ndarray:
    """Add detected staffs, staff assignments and inferred pitches to overlay."""
    overlay = primitive_overlay.copy()
    height, width = overlay.shape[:2]
    colors = ((40, 200, 40), (40, 130, 240))
    for staff in staffs:
        color = colors[staff.track % len(colors)]
        for y in staff.lines:
            cv2.line(overlay, (0, int(round(y))), (width - 1, int(round(y))), color, 1)
        label = f"S{staff.index + 1} {staff.clef}"
        cv2.putText(
            overlay,
            label,
            (12, max(70, min(height - 8, int(round(staff.lines[0] - 5))))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    for event in events:
        staff = staffs[event.staff_index]
        y = int(round(np.mean([staff.lines[-1] - (
            (_DIATONIC_STEPS.index(pitch[0]) + 7 * int(pitch[-1])) -
            (_DIATONIC_STEPS.index("E" if staff.clef == "treble" else "G") + 7 * (4 if staff.clef == "treble" else 2))
        ) * staff.spacing / 2.0 for pitch in event.pitches])))
        cv2.putText(
            overlay,
            "/".join(event.pitches),
            (int(round(event.x + 5)), max(12, min(height - 4, y - 5))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (30, 30, 220),
            1,
            cv2.LINE_AA,
        )
    return overlay


def convert_detection_to_musicxml(
    image: np.ndarray,
    detection: dict[str, Any],
    output_path: Path,
    overlay_path: Path,
    primitive_overlay: np.ndarray,
    staff_mode: str = "piano",
    beats: int = 4,
    beat_type: int = 4,
) -> dict[str, Any]:
    staffs = detect_staffs(image, float(detection["staff_spacing"]), staff_mode)
    events, event_warnings = build_note_events(image, detection, staffs)
    if not events:
        raise ValueError("No detected noteheads could be assigned to a staff")
    write_musicxml(events, staffs, output_path, Path(detection["source"]).stem, beats, beat_type)
    annotated = draw_musicxml_overlay(image, primitive_overlay, staffs, events)
    if not cv2.imwrite(str(overlay_path), annotated):
        raise RuntimeError(f"Failed to write {overlay_path}")
    warnings = [
        "Clefs are assumed from --staff-mode (piano defaults to alternating treble/bass).",
        "Key signatures are not detected; only an accidental printed directly next to a note is applied.",
        "Accidental type (sharp/flat/natural) is guessed from stroke geometry with no trained model or "
        "labelled dataset behind it -- verify every altered pitch before use.",
        "A detected accidental is applied only to the note it is attached to, not carried forward to "
        "later notes at the same staff position in the same measure (no reliable barline detection yet).",
        "Rests, dots, ties, tuplets, voices and printed time signatures are not detected yet.",
        "Measure boundaries are provisional: systems and accumulated duration are used instead of detected barlines.",
    ]
    warnings.extend(event_warnings)
    return {
        "schema_version": "1.0",
        "source": detection.get("source"),
        "staff_mode": staff_mode,
        "time_signature_assumption": f"{beats}/{beat_type}",
        "staffs": [asdict(staff) for staff in staffs],
        "events": [asdict(event) for event in events],
        "counts": {
            "staffs": len(staffs),
            "events": len(events),
            "pitches": sum(len(event.pitches) for event in events),
            "accidentals_detected": len(detection.get("accidentals", [])),
        },
        "outputs": {"musicxml": str(output_path), "overlay": str(overlay_path)},
        "warnings": warnings,
    }
