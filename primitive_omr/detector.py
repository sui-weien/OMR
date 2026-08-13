from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    notehead_threshold: float = 0.28
    stem_threshold: float = 0.22
    flag_threshold: float = 0.3
    flag_classifier_threshold: float = 0.75
    flag_fallback_threshold: float = 0.9
    target_pixels: int = 3_750_000
    min_staff_spacing: float = 6.0
    max_staff_spacing: float = 40.0


def _bbox_iou(first: list[int], second: list[int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(first_area + second_area - intersection)


def _bbox_smaller_overlap(first: list[int], second: list[int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(min(first_area, second_area))


def _continuous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def estimate_staff_spacing(gray: np.ndarray, config: DetectionConfig) -> float:
    """Estimate adjacent staff-line distance from long horizontal ink runs."""
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    width = gray.shape[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, width // 45), 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    margin = max(1, width // 18)
    projection = np.count_nonzero(horizontal[:, margin : width - margin], axis=1)
    threshold = max(12, int((width - 2 * margin) * 0.09))
    rows = _continuous_runs(projection >= threshold)
    centers = np.asarray([(start + end - 1) / 2 for start, end in rows], dtype=float)

    if centers.size >= 2:
        differences = np.diff(centers)
        valid = differences[
            (differences >= config.min_staff_spacing)
            & (differences <= config.max_staff_spacing)
        ]
        if valid.size:
            rounded = np.rint(valid).astype(int)
            histogram = np.bincount(rounded)
            mode = int(np.argmax(histogram))
            near_mode = valid[np.abs(valid - mode) <= 2]
            if near_mode.size:
                return float(np.median(near_mode))

    # Conservative fallback for 200-DPI pages used by the Xia dataset.
    return 12.0


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        boxes.append((int(x), int(y), int(x + width), int(y + height), int(area)))
    return boxes


def _split_notehead_box(
    box: tuple[int, int, int, int, int], spacing: float
) -> list[list[int]]:
    x0, y0, x1, y1, _ = box
    width = x1 - x0
    height = y1 - y0
    expected_width = max(4.0, spacing * 1.25)
    expected_height = max(4.0, spacing)
    columns = max(1, int(round(width / expected_width)))
    rows = max(1, int(round(height / expected_height)))
    if width < spacing * 1.8:
        columns = 1
    if height < spacing * 1.65:
        rows = 1
    columns = min(columns, 8)
    rows = min(rows, 12)
    result: list[list[int]] = []
    for row in range(rows):
        for column in range(columns):
            left = round(x0 + column * width / columns)
            right = round(x0 + (column + 1) * width / columns)
            top = round(y0 + row * height / rows)
            bottom = round(y0 + (row + 1) * height / rows)
            result.append([left, top, right, bottom])
    return result


def detect_noteheads(
    note_score: np.ndarray, spacing: float, threshold: float
) -> list[dict[str, Any]]:
    mask = (note_score >= threshold).astype(np.uint8)
    close_width = max(2, int(round(spacing * 0.22)))
    close_height = max(2, int(round(spacing * 0.18)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_width, close_height)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    candidates: list[dict[str, Any]] = []
    for component in _component_boxes(mask):
        x0, y0, x1, y1, area = component
        width = x1 - x0
        height = y1 - y0
        if width < spacing * 0.28 or height < spacing * 0.32:
            continue
        if width > spacing * 9 or height > spacing * 13:
            continue
        if area < spacing * spacing * 0.12:
            continue
        for bbox in _split_notehead_box(component, spacing):
            bx0, by0, bx1, by1 = bbox
            region = note_score[by0:by1, bx0:bx1]
            if region.size == 0:
                continue
            positive = region[region >= threshold]
            if positive.size < spacing * spacing * 0.08:
                continue
            confidence = float(np.mean(positive))
            candidates.append(
                {
                    "bbox": bbox,
                    "center": [round((bx0 + bx1) / 2, 2), round((by0 + by1) / 2, 2)],
                    "confidence": round(confidence, 4),
                }
            )

    candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        duplicate = next(
            (
                existing
                for existing in deduplicated
                if _bbox_iou(candidate["bbox"], existing["bbox"]) > 0.55
            ),
            None,
        )
        if duplicate is None:
            deduplicated.append(candidate)
        elif candidate["confidence"] > duplicate["confidence"]:
            deduplicated.remove(duplicate)
            deduplicated.append(candidate)
    # The segmentation model's notehead channel is most confident near the
    # center of a printed notehead; thresholding it alone reports a box
    # noticeably smaller than the glyph's real visual extent (confirmed
    # against ground-truth annotation boxes). Grow the box a bit rather
    # than lowering the acceptance threshold globally, which would also
    # accept many new, weaker candidates elsewhere on the page.
    # detect_stems' proximity thresholds were tuned against the
    # pre-growth box, so keep it under "core_bbox" for that one consumer;
    # everything else (overlay drawing, accidental search anchoring,
    # beam-mask blanking, ground-truth comparison) should see the
    # visually-accurate grown box in "bbox".
    margin_x = max(1, int(round(spacing * 0.08)))
    margin_y = max(1, int(round(spacing * 0.08)))
    height_limit, width_limit = note_score.shape
    for index, item in enumerate(deduplicated):
        item["id"] = f"notehead-{index}"
        x0, y0, x1, y1 = item["bbox"]
        item["core_bbox"] = item["bbox"]
        item["bbox"] = [
            max(0, x0 - margin_x),
            max(0, y0 - margin_y),
            min(width_limit, x1 + margin_x),
            min(height_limit, y1 + margin_y),
        ]
    return deduplicated


def _merge_vertical_lines(
    lines: list[tuple[int, int, int, int]], spacing: float
) -> list[list[int]]:
    if not lines:
        return []
    normalized = []
    for x0, y0, x1, y1 in lines:
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        normalized.append([int(round((x0 + x1) / 2)), y0, y1])
    normalized.sort(key=lambda item: (item[0], item[1]))

    merged: list[list[int]] = []
    for x, top, bottom in normalized:
        match = None
        for existing in merged:
            # Hough commonly returns the left edge, center, and right edge of
            # the same printed stem.  Their separation is much smaller than
            # the horizontal distance between neighboring musical stems.
            x_close = abs(existing[0] - x) <= max(3, spacing * 0.58)
            y_close = top <= existing[2] + spacing * 0.8 and bottom >= existing[1] - spacing * 0.8
            if x_close and y_close:
                match = existing
                break
        if match is None:
            merged.append([x, top, bottom])
        else:
            match[0] = int(round((match[0] + x) / 2))
            match[1] = min(match[1], top)
            match[2] = max(match[2], bottom)
    return [[x, top, x, bottom] for x, top, bottom in merged]


def detect_stems(
    gray: np.ndarray,
    stem_score: np.ndarray,
    noteheads: list[dict[str, Any]],
    spacing: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    score_mask = (stem_score >= threshold).astype(np.uint8) * 255
    vertical_length = max(5, int(round(spacing * 1.05)))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    raw_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    score_vertical = cv2.morphologyEx(score_mask, cv2.MORPH_OPEN, vertical_kernel)
    candidate = cv2.bitwise_or(raw_vertical, score_vertical)

    detected = cv2.HoughLinesP(
        candidate,
        1,
        np.pi / 180,
        threshold=max(8, int(round(spacing * 0.9))),
        minLineLength=max(8, int(round(spacing * 1.2))),
        maxLineGap=max(2, int(round(spacing * 0.65))),
    )
    raw_lines: list[tuple[int, int, int, int]] = []
    if detected is not None:
        for line in detected[:, 0, :]:
            x0, y0, x1, y1 = (int(value) for value in line)
            dy = abs(y1 - y0)
            dx = abs(x1 - x0)
            if dy < spacing * 1.15 or dx > max(3, dy * 0.16):
                continue
            raw_lines.append((x0, y0, x1, y1))

    merged_lines = _merge_vertical_lines(raw_lines, spacing)
    stems: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    for line in merged_lines:
        x, top, _, bottom = line
        linked_notes: list[tuple[float, dict[str, Any]]] = []
        for note in noteheads:
            nx0, ny0, nx1, ny1 = note.get("core_bbox", note["bbox"])
            horizontal_gap = 0.0
            if x < nx0:
                horizontal_gap = nx0 - x
            elif x > nx1:
                horizontal_gap = x - nx1
            vertical_overlap = bottom >= ny0 - spacing * 0.7 and top <= ny1 + spacing * 0.7
            if horizontal_gap <= spacing * 0.48 and vertical_overlap:
                center_y = note["center"][1]
                endpoint_gap = min(abs(center_y - top), abs(center_y - bottom))
                score = horizontal_gap + endpoint_gap * 0.08
                linked_notes.append((score, note))

        if not linked_notes:
            continue
        linked_notes.sort(key=lambda item: item[0])
        region = stem_score[max(0, top) : min(stem_score.shape[0], bottom + 1), max(0, x - 1) : x + 2]
        model_coverage = float(np.mean(region >= threshold)) if region.size else 0.0
        mean_probability = float(np.mean(region)) if region.size else 0.0
        source = "model+geometry" if model_coverage >= 0.1 else "geometry"
        endpoint_near_note = any(
            min(abs(note["center"][1] - top), abs(note["center"][1] - bottom))
            <= spacing * 1.05
            for _, note in linked_notes
        )
        # A real stem terminates at a notehead. Barlines and character strokes
        # may pass close to noteheads but usually continue through the staff.
        if not endpoint_near_note:
            continue
        if source == "geometry":
            if bottom - top > spacing * 5.5:
                continue
        stem_id = f"stem-{len(stems)}"
        confidence = max(mean_probability, min(1.0, 0.35 + model_coverage))
        stems.append(
            {
                "id": stem_id,
                "line": line,
                "length": round(float(bottom - top), 2),
                "confidence": round(confidence, 4),
                "model_coverage": round(model_coverage, 4),
                "mean_probability": round(mean_probability, 4),
                "source": source,
            }
        )
        best_score = linked_notes[0][0]
        for score, note in linked_notes:
            if score <= best_score + spacing * 0.55:
                links.append({"notehead_id": note["id"], "stem_id": stem_id})

    # Recover short stems that the full-page Hough transform misses. Search
    # only beside an otherwise-unlinked notehead, which keeps text/barline
    # false positives much lower than globally reducing the Hough threshold.
    linked_note_ids = {relation["notehead_id"] for relation in links}
    for note in noteheads:
        if note["id"] in linked_note_ids:
            continue
        nx0, ny0, nx1, ny1 = note.get("core_bbox", note["bbox"])
        search_x0 = max(0, int(round(nx0 - spacing * 0.45)))
        search_x1 = min(candidate.shape[1], int(round(nx1 + spacing * 0.45)) + 1)
        search_y0 = max(0, int(round(ny0 - spacing * 5.0)))
        search_y1 = min(candidate.shape[0], int(round(ny1 + spacing * 5.0)) + 1)
        options: list[tuple[float, int, int, int]] = []
        for x in range(search_x0, search_x1):
            column = candidate[search_y0:search_y1, x] > 0
            for local_top, local_bottom in _continuous_runs(column):
                top = search_y0 + local_top
                bottom = search_y0 + local_bottom - 1
                length = bottom - top
                if length < spacing * 1.15 or length > spacing * 5.5:
                    continue
                overlaps_note = bottom >= ny0 - spacing * 0.45 and top <= ny1 + spacing * 0.45
                extends_from_note = top <= ny0 - spacing * 0.5 or bottom >= ny1 + spacing * 0.5
                endpoint_gap = min(
                    abs(note["center"][1] - top),
                    abs(note["center"][1] - bottom),
                )
                if not overlaps_note or not extends_from_note or endpoint_gap > spacing * 1.2:
                    continue
                edge_distance = min(abs(x - nx0), abs(x - nx1))
                if edge_distance > spacing * 0.55:
                    continue
                probability_region = stem_score[
                    max(0, top) : min(stem_score.shape[0], bottom + 1),
                    max(0, x - 1) : min(stem_score.shape[1], x + 2),
                ]
                mean_probability = (
                    float(np.mean(probability_region)) if probability_region.size else 0.0
                )
                score = length + mean_probability * spacing * 2.0 - edge_distance
                options.append((score, x, top, bottom))
        if not options:
            continue
        _, x, top, bottom = max(options, key=lambda item: item[0])
        existing = next(
            (
                stem
                for stem in stems
                if abs(stem["line"][0] - x) <= spacing * 0.55
                and stem["line"][3] >= top - spacing * 0.5
                and stem["line"][1] <= bottom + spacing * 0.5
            ),
            None,
        )
        if existing is not None:
            links.append({"notehead_id": note["id"], "stem_id": existing["id"]})
            linked_note_ids.add(note["id"])
            continue
        region = stem_score[
            max(0, top) : min(stem_score.shape[0], bottom + 1),
            max(0, x - 1) : min(stem_score.shape[1], x + 2),
        ]
        model_coverage = float(np.mean(region >= threshold)) if region.size else 0.0
        mean_probability = float(np.mean(region)) if region.size else 0.0
        stem_id = f"stem-{len(stems)}"
        stems.append(
            {
                "id": stem_id,
                "line": [x, top, x, bottom],
                "length": round(float(bottom - top), 2),
                "confidence": round(max(0.4, mean_probability), 4),
                "model_coverage": round(model_coverage, 4),
                "mean_probability": round(mean_probability, 4),
                "source": "note-guided-recovery",
            }
        )
        links.append({"notehead_id": note["id"], "stem_id": stem_id})
        linked_note_ids.add(note["id"])
    return stems, links


def _mask_rect(mask: np.ndarray, bbox: list[int], padding: int) -> None:
    x0, y0, x1, y1 = bbox
    cv2.rectangle(
        mask,
        (max(0, x0 - padding), max(0, y0 - padding)),
        (min(mask.shape[1] - 1, x1 + padding), min(mask.shape[0] - 1, y1 + padding)),
        0,
        thickness=-1,
    )


def _thick_bridge_coverage(
    binary: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    spacing: float,
) -> float:
    """Measure how much of the line between two stem endpoints is beam-thick."""
    x0, y0 = first
    x1, y1 = second
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length = float(np.hypot(dx, dy))
    if length < 2:
        return 0.0
    sample_count = max(3, int(round(length)) + 1)
    radius = max(3, int(round(spacing * 0.65)))
    along = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    perpendicular_x = -dy / length
    perpendicular_y = dx / length
    center_x = x0 + along * dx
    center_y = y0 + along * dy
    map_x = center_x[None, :] + offsets[:, None] * perpendicular_x
    map_y = center_y[None, :] + offsets[:, None] * perpendicular_y
    sampled = cv2.remap(
        binary,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ink = sampled > 0
    longest_runs = np.zeros(sample_count, dtype=np.int32)
    for column_index in range(sample_count):
        runs = _continuous_runs(ink[:, column_index])
        if runs:
            longest_runs[column_index] = max(end - start for start, end in runs)
    # Count one contiguous thick stroke, rather than summing separate staff,
    # slur, and notehead pixels within the same sampling column.
    beam_columns = longest_runs >= max(4, int(round(spacing * 0.28)))
    endpoint_margin = min(sample_count // 4, max(1, int(round(spacing * 0.25))))
    interior = beam_columns[endpoint_margin : sample_count - endpoint_margin]
    return float(np.mean(interior)) if interior.size else 0.0


def _parallel_bridge_bands(
    binary: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    spacing: float,
) -> list[tuple[float, float, float]]:
    """Return perpendicular offset, thickness, and coverage for parallel beams."""
    x0, y0 = first
    x1, y1 = second
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length = float(np.hypot(dx, dy))
    if length < 2:
        return []
    sample_count = max(3, int(round(length)) + 1)
    radius = max(5, int(round(spacing * 1.8)))
    along = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    perpendicular_x = -dy / length
    perpendicular_y = dx / length
    center_x = x0 + along * dx
    center_y = y0 + along * dy
    map_x = center_x[None, :] + offsets[:, None] * perpendicular_x
    map_y = center_y[None, :] + offsets[:, None] * perpendicular_y
    sampled = cv2.remap(
        binary,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    margin = min(sample_count // 4, max(1, int(round(spacing * 0.3))))
    interior = sampled[:, margin : sample_count - margin]
    if interior.size == 0:
        return []
    coverage_by_offset = np.mean(interior > 0, axis=1)
    active_runs = _continuous_runs(coverage_by_offset >= 0.58)
    bands: list[tuple[float, float, float]] = []
    for start, end in active_runs:
        thickness = end - start
        if thickness < max(3, int(round(spacing * 0.22))):
            continue
        offset = float(np.mean(offsets[start:end]))
        if abs(offset) > spacing * 1.45:
            continue
        bands.append((offset, float(thickness), float(np.mean(coverage_by_offset[start:end]))))
    bands.sort(key=lambda item: abs(item[0]))
    return bands[:4]


def detect_beams(
    gray: np.ndarray,
    noteheads: list[dict[str, Any]],
    stems: list[dict[str, Any]],
    notehead_stem_links: list[dict[str, str]],
    spacing: float,
    beam_score: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    work = binary.copy()
    if beam_score is not None:
        probability_mask = (beam_score >= 0.22).astype(np.uint8) * 255
        work = cv2.bitwise_or(work, probability_mask)
    for note in noteheads:
        _mask_rect(work, note["bbox"], max(1, int(round(spacing * 0.12))))
    for stem in stems:
        x0, y0, x1, y1 = stem["line"]
        cv2.line(
            work,
            (x0, y0),
            (x1, y1),
            0,
            thickness=max(2, int(round(spacing * 0.28))),
        )

    distance = cv2.distanceTransform((work > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thick = (distance >= max(1.2, spacing * 0.12)).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(spacing * 0.75))), max(1, int(round(spacing * 0.12)))),
    )
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    note_by_id = {note["id"]: note for note in noteheads}
    notes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for relation in notehead_stem_links:
        note = note_by_id.get(relation["notehead_id"])
        if note is not None:
            notes_by_stem.setdefault(relation["stem_id"], []).append(note)

    beam_endpoint_by_stem: dict[str, tuple[int, int]] = {}
    for stem in stems:
        x, top, _, bottom = stem["line"]
        linked_notes = notes_by_stem.get(stem["id"], [])
        if not linked_notes:
            continue
        note_center_y = float(np.median([note["center"][1] for note in linked_notes]))
        beam_y = top if abs(note_center_y - top) > abs(note_center_y - bottom) else bottom
        beam_endpoint_by_stem[stem["id"]] = (x, beam_y)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        rectangle = cv2.minAreaRect(contour)
        (_, _), (width, height), angle = rectangle
        long_side = max(width, height)
        short_side = min(width, height)
        if short_side <= 0:
            continue
        # A partial/broken beam fragment on an isolated subdivided note can
        # print noticeably thinner than a full multi-note beam. The floor
        # here used to sit above some real fragments' actual ink thickness,
        # silently dropping them before they ever reached the stem-adjacency
        # check below, which is this function's main defense against thin
        # staff-line or slur remnants becoming false beam candidates.
        if long_side < spacing * 1.0 or short_side < spacing * 0.11:
            continue
        if short_side > spacing * 1.15 or long_side / short_side < 1.7:
            continue
        rectangle_area = max(1.0, long_side * short_side)
        fill_ratio = float(cv2.contourArea(contour) / rectangle_area)
        # A beam is a compact filled quadrilateral. Slurs and ties can have a
        # similar oriented bounding rectangle, but occupy only a thin curved
        # fraction of that rectangle.
        long_axis_angle = angle if width >= height else angle + 90.0
        while long_axis_angle > 90:
            long_axis_angle -= 180
        while long_axis_angle <= -90:
            long_axis_angle += 180
        if abs(long_axis_angle) > 38:
            continue
        box_points = np.int32(np.round(cv2.boxPoints(rectangle)))
        x, y, box_width, box_height = cv2.boundingRect(box_points)
        bbox = [int(x), int(y), int(x + box_width), int(y + box_height)]
        candidates.append(
            {
                "bbox": bbox,
                "polygon": box_points.tolist(),
                "angle": round(float(long_axis_angle), 2),
                "length": round(float(long_side), 2),
                "thickness": round(float(short_side * 2), 2),
                "fill_ratio": round(fill_ratio, 4),
            }
        )

    beams: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["bbox"]
        linked_stems: list[str] = []
        for stem in stems:
            endpoint = beam_endpoint_by_stem.get(stem["id"])
            if endpoint is None:
                continue
            sx, sy = endpoint
            endpoint_near = (
                x0 - spacing * 0.65 <= sx <= x1 + spacing * 0.65
                and y0 - spacing * 0.8 <= sy <= y1 + spacing * 0.8
            )
            if endpoint_near:
                linked_stems.append(stem["id"])
        if not linked_stems:
            continue
        fill_ratio = candidate["fill_ratio"]
        high_confidence_shape = fill_ratio >= 0.42
        if len(linked_stems) == 1:
            # A one-stem beam is a short partial beam/flag fragment -- for
            # example the broken half-beam notation on an isolated
            # subdivided note that isn't grouped with a neighbour. Longer or
            # thicker one-stem rectangles are commonly text strokes near a
            # stem endpoint instead (for example tempo markings), so gate on
            # shape size rather than requiring the same fill ratio used for
            # full multi-note beams below: a short, thin, stem-anchored
            # fragment is already unlikely to be text.
            if candidate["length"] > spacing * 1.8 or candidate["thickness"] > spacing * 1.5:
                continue
            if fill_ratio < 0.28:
                continue
        else:
            multi_stem_recovery = fill_ratio >= 0.2
            if not (high_confidence_shape or multi_stem_recovery):
                continue
        beam_id = f"beam-{len(beams)}"
        candidate["id"] = beam_id
        if high_confidence_shape:
            candidate["source"] = "shape+stem"
        elif len(linked_stems) == 1:
            candidate["source"] = "single-stem-partial-beam"
        else:
            candidate["source"] = "multi-stem-recovery"
        candidate["linked_stem_count"] = len(linked_stems)
        candidate["confidence"] = round(
            min(1.0, 0.25 + fill_ratio * 0.55 + len(linked_stems) * 0.1), 4
        )
        beams.append(candidate)
        links.extend({"stem_id": stem_id, "beam_id": beam_id} for stem_id in linked_stems)

    # Recover beams missed by contour extraction. Two neighboring stem
    # endpoints are joined only when the pixels between them remain thick for
    # most of the path; one- or two-pixel staff/slur lines therefore fail.
    endpoint_items = sorted(beam_endpoint_by_stem.items(), key=lambda item: item[1][0])
    adjacency: dict[str, set[str]] = {stem_id: set() for stem_id, _ in endpoint_items}
    bridge_scores: dict[tuple[str, str], float] = {}
    for index, (first_id, first_point) in enumerate(endpoint_items):
        for second_id, second_point in endpoint_items[index + 1 :]:
            dx = second_point[0] - first_point[0]
            if dx > spacing * 4.2:
                break
            if dx < spacing * 0.55 or abs(second_point[1] - first_point[1]) > spacing * 0.85:
                continue
            coverage = _thick_bridge_coverage(binary, first_point, second_point, spacing)
            if coverage < 0.68:
                continue
            adjacency[first_id].add(second_id)
            adjacency[second_id].add(first_id)
            bridge_scores[tuple(sorted((first_id, second_id)))] = coverage

    visited: set[str] = set()
    existing_link_pairs = {(item["stem_id"], item["beam_id"]) for item in links}
    for stem_id in adjacency:
        if stem_id in visited or not adjacency[stem_id]:
            continue
        stack = [stem_id]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) < 2:
            continue
        existing_beam_ids = {
            beam_id
            for linked_stem_id, beam_id in existing_link_pairs
            if linked_stem_id in component
        }
        if any(
            sum((linked_stem_id, beam_id) in existing_link_pairs for linked_stem_id in component) >= 2
            for beam_id in existing_beam_ids
        ):
            continue
        already_linked_stems = {
            linked_stem_id
            for linked_stem_id, _ in existing_link_pairs
            if linked_stem_id in component
        }
        if len(already_linked_stems) >= 2:
            continue
        points = [beam_endpoint_by_stem[item] for item in component]
        points.sort()
        first_point = points[0]
        last_point = points[-1]
        dx = max(1, last_point[0] - first_point[0])
        slope = (last_point[1] - first_point[1]) / dx
        edge_coverages = [
            score
            for pair, score in bridge_scores.items()
            if pair[0] in component and pair[1] in component
        ]
        length = float(np.hypot(dx, last_point[1] - first_point[1]))
        perpendicular_x = -(last_point[1] - first_point[1]) / max(1.0, length)
        perpendicular_y = dx / max(1.0, length)
        bands = _parallel_bridge_bands(binary, first_point, last_point, spacing)
        if not bands:
            bands = [(0.0, max(4.0, spacing * 0.5), float(np.mean(edge_coverages)))]
        group_id = f"beam-group-{len(beams)}"
        for offset, thickness, band_coverage in bands:
            half_thickness = max(2.0, thickness / 2.0)
            shifted_first = (
                first_point[0] + offset * perpendicular_x,
                first_point[1] + offset * perpendicular_y,
            )
            shifted_last = (
                last_point[0] + offset * perpendicular_x,
                last_point[1] + offset * perpendicular_y,
            )
            polygon_float = [
                [shifted_first[0] - half_thickness * perpendicular_x, shifted_first[1] - half_thickness * perpendicular_y],
                [shifted_last[0] - half_thickness * perpendicular_x, shifted_last[1] - half_thickness * perpendicular_y],
                [shifted_last[0] + half_thickness * perpendicular_x, shifted_last[1] + half_thickness * perpendicular_y],
                [shifted_first[0] + half_thickness * perpendicular_x, shifted_first[1] + half_thickness * perpendicular_y],
            ]
            polygon = [[int(round(x)), int(round(y))] for x, y in polygon_float]
            x_values = [point[0] for point in polygon]
            y_values = [point[1] for point in polygon]
            bbox = [min(x_values), min(y_values), max(x_values) + 1, max(y_values) + 1]
            if any(
                _bbox_smaller_overlap(bbox, existing_beam["bbox"]) >= 0.55
                for existing_beam in beams
            ):
                continue
            beam_id = f"beam-{len(beams)}"
            beams.append(
                {
                    "id": beam_id,
                    "group_id": group_id,
                    "bbox": bbox,
                    "polygon": polygon,
                    "angle": round(float(np.degrees(np.arctan(slope))), 2),
                    "length": round(length, 2),
                    "thickness": round(float(thickness), 2),
                    "fill_ratio": round(band_coverage, 4),
                    "source": "stem-bridge-recovery",
                    "linked_stem_count": len(component),
                    "confidence": round(min(1.0, 0.35 + band_coverage * 0.55), 4),
                }
            )
            for linked_stem_id in component:
                links.append({"stem_id": linked_stem_id, "beam_id": beam_id})
    return beams, links


def detect_segmented_flags(
    flag_score: np.ndarray,
    noteheads: list[dict[str, Any]],
    stems: list[dict[str, Any]],
    notehead_stem_links: list[dict[str, str]],
    stem_beam_links: list[dict[str, str]],
    spacing: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Convert a directly trained flag probability channel into linked flags."""
    mask = (flag_score >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    note_by_id = {note["id"]: note for note in noteheads}
    notes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for relation in notehead_stem_links:
        note = note_by_id.get(relation["notehead_id"])
        if note is not None:
            notes_by_stem.setdefault(relation["stem_id"], []).append(note)
    stems_with_beams = {relation["stem_id"] for relation in stem_beam_links}

    flags: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    for x0, y0, x1, y1, area in _component_boxes(mask):
        width = x1 - x0
        height = y1 - y0
        if width < spacing * 0.2 or height < spacing * 0.35:
            continue
        if width > spacing * 2.8 or height > spacing * 5.0:
            continue
        if area < spacing * spacing * 0.04:
            continue
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        candidates: list[tuple[float, dict[str, Any], bool]] = []
        for stem in stems:
            if stem["id"] in stems_with_beams:
                continue
            linked_notes = notes_by_stem.get(stem["id"], [])
            if not linked_notes:
                continue
            stem_x, top, _, bottom = stem["line"]
            note_ys = [note["center"][1] for note in linked_notes]
            distance_to_top = min(abs(value - top) for value in note_ys)
            distance_to_bottom = min(abs(value - bottom) for value in note_ys)
            stem_up = distance_to_bottom < distance_to_top
            endpoint_y = top if stem_up else bottom
            if stem_up:
                directional = (
                    center_x >= stem_x - spacing * 0.3
                    and center_x <= stem_x + spacing * 2.4
                    and center_y >= endpoint_y - spacing * 0.5
                    and center_y <= endpoint_y + spacing * 3.8
                )
            else:
                directional = (
                    center_x <= stem_x + spacing * 0.3
                    and center_x >= stem_x - spacing * 2.4
                    and center_y <= endpoint_y + spacing * 0.5
                    and center_y >= endpoint_y - spacing * 3.8
                )
            if not directional:
                continue
            gap_x = max(x0 - stem_x, stem_x - x1, 0)
            gap_y = max(y0 - endpoint_y, endpoint_y - y1, 0)
            distance = float(np.hypot(gap_x, gap_y))
            if distance <= spacing * 1.25:
                candidates.append((distance, stem, stem_up))
        if not candidates:
            continue
        _, stem, stem_up = min(candidates, key=lambda item: item[0])
        region_score = flag_score[y0:y1, x0:x1]
        confidence = float(np.mean(region_score[mask[y0:y1, x0:x1] > 0]))
        bbox = [x0, y0, x1, y1]
        flag_id = f"flag-{len(flags)}"
        flags.append(
            {
                "id": flag_id,
                "bbox": bbox,
                "count": max(1, min(4, int(round(height / max(1.0, spacing * 1.55))))),
                "direction": "down-right" if stem_up else "up-left",
                "confidence": round(confidence, 4),
                "source": "direct-flag-segmentation",
                "topology": "model-mask",
                "classifier": None,
            }
        )
        links.append({"stem_id": stem["id"], "flag_id": flag_id})
    return flags, links


def detect_flags(
    gray: np.ndarray,
    noteheads: list[dict[str, Any]],
    stems: list[dict[str, Any]],
    notehead_stem_links: list[dict[str, str]],
    stem_beam_links: list[dict[str, str]],
    spacing: float,
    flag_classifier: Any | None = None,
    classifier_threshold: float = 0.75,
    fallback_classifier_threshold: float = 0.9,
    flag_score: np.ndarray | None = None,
    flag_threshold: float = 0.3,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Detect curved single-stem flags at the endpoint opposite the notehead."""
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    staff_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, int(round(spacing * 3.0))), 1)
    )
    long_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, staff_kernel)
    work = cv2.bitwise_and(binary, cv2.bitwise_not(long_horizontal))

    note_by_id = {note["id"]: note for note in noteheads}
    notes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for relation in notehead_stem_links:
        note = note_by_id.get(relation["notehead_id"])
        if note is not None:
            notes_by_stem.setdefault(relation["stem_id"], []).append(note)
    stems_with_beams = {relation["stem_id"] for relation in stem_beam_links}

    if flag_score is not None:
        flags, links = detect_segmented_flags(
            flag_score,
            noteheads,
            stems,
            notehead_stem_links,
            stem_beam_links,
            spacing,
            flag_threshold,
        )
    else:
        flags, links = [], []
    directly_linked_stems = {relation["stem_id"] for relation in links}
    for stem in stems:
        if stem["id"] in stems_with_beams or stem["id"] in directly_linked_stems:
            continue
        linked_notes = notes_by_stem.get(stem["id"], [])
        if not linked_notes:
            continue
        x, top, _, bottom = stem["line"]
        note_center_ys = [note["center"][1] for note in linked_notes]
        distance_to_top = min(abs(center_y - top) for center_y in note_center_ys)
        distance_to_bottom = min(abs(center_y - bottom) for center_y in note_center_ys)
        near_distance = min(distance_to_top, distance_to_bottom)
        far_distance = max(distance_to_top, distance_to_bottom)
        if far_distance < spacing * 1.5 or far_distance - near_distance < spacing * 0.75:
            continue
        stem_up = distance_to_bottom < distance_to_top
        endpoint_y = top if stem_up else bottom
        classifier_result = None
        if flag_classifier is not None:
            classifier_result = flag_classifier.validate(
                gray, stem, linked_notes, stem_up
            )
            if (
                not classifier_result["accepted"]
                or classifier_result["confidence"] < classifier_threshold
            ):
                continue
        horizontal_reach = max(6, int(round(spacing * 1.8)))
        vertical_reach = max(8, int(round(spacing * 2.4)))
        if stem_up:
            crop_x0 = max(0, x - round(spacing * 0.15))
            crop_x1 = min(work.shape[1], x + horizontal_reach)
            crop_y0 = max(0, endpoint_y - round(spacing * 0.3))
            crop_y1 = min(work.shape[0], endpoint_y + vertical_reach)
        else:
            crop_x0 = max(0, x - horizontal_reach)
            crop_x1 = min(work.shape[1], x + round(spacing * 0.15))
            crop_y0 = max(0, endpoint_y - vertical_reach)
            crop_y1 = min(work.shape[0], endpoint_y + round(spacing * 0.3))
        crop = work[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        if crop.size == 0:
            continue
        local_stem_x = x - crop_x0
        cv2.line(
            crop,
            (local_stem_x, 0),
            (local_stem_x, crop.shape[0] - 1),
            0,
            max(1, int(round(spacing * 0.18))),
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(2, int(round(spacing * 0.18))), max(2, int(round(spacing * 0.18)))),
        )
        crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, close_kernel)
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            (crop > 0).astype(np.uint8), connectivity=8
        )
        local_endpoint_y = endpoint_y - crop_y0
        contact_x_radius = max(2, int(round(spacing * 0.55)))
        contact_y_before = max(1, int(round(spacing * 0.25)))
        # Removing the stem can leave a small antialiasing gap before the
        # curved tail begins, so keep the endpoint seed band slightly deeper.
        contact_y_after = max(2, int(round(spacing * 0.6)))
        if stem_up:
            contact_x0 = min(crop.shape[1], local_stem_x + 1)
            contact_x1 = min(crop.shape[1], local_stem_x + contact_x_radius)
            contact_y0 = max(0, local_endpoint_y - contact_y_before)
            contact_y1 = min(crop.shape[0], local_endpoint_y + contact_y_after)
        else:
            contact_x0 = max(0, local_stem_x - contact_x_radius)
            contact_x1 = max(0, local_stem_x)
            contact_y0 = max(0, local_endpoint_y - contact_y_after)
            contact_y1 = min(crop.shape[0], local_endpoint_y + contact_y_before)
        if contact_x1 <= contact_x0 or contact_y1 <= contact_y0:
            continue
        touching_labels = set(
            int(label)
            for label in np.unique(
                component_labels[contact_y0:contact_y1, contact_x0:contact_x1]
            )
            if label != 0
        )
        endpoint_connected_labels = touching_labels
        # The 200-DPI scans often contain a 1–3 px antialiasing gap between
        # the stem and the curved tail.  Once the rhythm CNN strongly supports
        # n816, also inspect nearby components instead of requiring literal
        # pixel connectivity.
        if classifier_result is not None:
            component_indices = range(1, component_count)
        else:
            component_indices = endpoint_connected_labels
        if not endpoint_connected_labels and classifier_result is None:
            continue
        candidates: list[tuple[float, list[int], int]] = []
        candidate_connectivity: dict[tuple[int, int, int, int], bool] = {}
        for component_index in component_indices:
            if component_index >= component_count:
                continue
            cx0, cy0, width, height, area = component_stats[component_index]
            cx0 = int(cx0)
            cy0 = int(cy0)
            width = int(width)
            height = int(height)
            area = int(area)
            cx1 = cx0 + width
            cy1 = cy0 + height
            width = cx1 - cx0
            height = cy1 - cy0
            if width < spacing * 0.35 or height < spacing * 0.65:
                continue
            if width > spacing * 2.0 or height > spacing * 2.8:
                continue
            # A real flag is primarily a curved vertical tail.  Short,
            # horizontal components are usually fragments of a missed beam.
            if height < width * 0.75:
                continue
            if area < spacing * spacing * 0.06 or area > spacing * spacing * 2.2:
                continue
            fill_ratio = area / max(1.0, width * height)
            if fill_ratio > 0.72:
                # Filled noteheads and text blocks are much denser than a
                # curved flag stroke.
                continue
            global_bbox = [crop_x0 + cx0, crop_y0 + cy0, crop_x0 + cx1, crop_y0 + cy1]
            center_x = (global_bbox[0] + global_bbox[2]) / 2
            center_y = (global_bbox[1] + global_bbox[3]) / 2
            if stem_up:
                directional = center_x >= x + spacing * 0.15 and center_y >= endpoint_y + spacing * 0.3
                starts_at_endpoint = (
                    global_bbox[0] <= x + spacing * 0.55
                    and global_bbox[3] >= endpoint_y + spacing * 0.7
                )
            else:
                directional = center_x <= x - spacing * 0.15 and center_y <= endpoint_y - spacing * 0.3
                starts_at_endpoint = (
                    global_bbox[2] >= x - spacing * 0.55
                    and global_bbox[1] <= endpoint_y - spacing * 0.7
                )
            if not directional or not starts_at_endpoint:
                continue
            nearest_x = min(abs(global_bbox[0] - x), abs(global_bbox[2] - x))
            nearest_y = min(abs(global_bbox[1] - endpoint_y), abs(global_bbox[3] - endpoint_y))
            is_connected = component_index in endpoint_connected_labels
            max_x_gap = spacing * (0.55 if is_connected else 0.85)
            max_y_gap = spacing * (0.75 if is_connected else 1.0)
            if nearest_x > max_x_gap or nearest_y > max_y_gap:
                continue
            distance_score = nearest_x + nearest_y + (0.0 if is_connected else spacing * 0.35)
            candidates.append((distance_score, global_bbox, area))
            candidate_connectivity[tuple(global_bbox)] = is_connected
        inferred_from_cnn = False
        if candidates:
            _, bbox, area = min(candidates, key=lambda item: item[0])
            endpoint_connected = candidate_connectivity[tuple(bbox)]
        else:
            # The classifier was trained on the complete note+stem crop and
            # can still recognize n816 when a faint flag is disconnected or
            # merged into a slur.  Use a conservative inferred region only for
            # strong model-supported stems, and reject long horizontal ink at
            # the far endpoint because that indicates an undetected beam.
            if (
                classifier_result is None
                or classifier_result["confidence"] < fallback_classifier_threshold
                or stem.get("source") != "model+geometry"
                or stem.get("confidence", 0.0) < 0.65
            ):
                continue
            band_y0 = max(0, int(round(endpoint_y - spacing * 0.45)))
            band_y1 = min(gray.shape[0], int(round(endpoint_y + spacing * 0.45)) + 1)
            band_x0 = max(0, int(round(x - spacing * 2.2)))
            band_x1 = min(gray.shape[1], int(round(x + spacing * 2.2)) + 1)
            horizontal_band = long_horizontal[band_y0:band_y1, band_x0:band_x1]
            horizontal_row_lengths = (
                np.count_nonzero(horizontal_band, axis=1)
                if horizontal_band.size
                else np.empty(0, dtype=int)
            )
            thick_horizontal_rows = int(
                np.count_nonzero(horizontal_row_lengths >= spacing * 1.4)
            )
            # A staff line normally occupies only one or two rows.  Beams are
            # thick, so require several long rows before rejecting the flag.
            if thick_horizontal_rows >= max(3, int(round(spacing * 0.25))):
                continue
            if stem_up:
                bbox = [
                    max(0, int(round(x - spacing * 0.1))),
                    max(0, int(round(endpoint_y - spacing * 0.2))),
                    min(gray.shape[1], int(round(x + spacing * 1.35))),
                    min(gray.shape[0], int(round(endpoint_y + spacing * 1.9))),
                ]
            else:
                bbox = [
                    max(0, int(round(x - spacing * 1.35))),
                    max(0, int(round(endpoint_y - spacing * 1.9))),
                    min(gray.shape[1], int(round(x + spacing * 0.1))),
                    min(gray.shape[0], int(round(endpoint_y + spacing * 0.2))),
                ]
            # The classifier alone cannot tell a real flag from a blank
            # staff gap, a rest, or nearby text/slur ink: it only ever saw
            # isolated note+stem crops during training, and its input crop
            # is wide enough to pick up unrelated marks near the stem.
            # Require some actual ink to physically touch the stem tip
            # (the same contact zone used for the geometry-first path)
            # before trusting a CNN-only call.
            if not endpoint_connected_labels:
                continue
            area = 0
            endpoint_connected = False
            inferred_from_cnn = True
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        duplicate = any(
            _bbox_iou(bbox, existing["bbox"]) >= 0.25
            or (
                abs(bbox_center[0] - (existing["bbox"][0] + existing["bbox"][2]) / 2)
                <= spacing * 0.45
                and abs(bbox_center[1] - (existing["bbox"][1] + existing["bbox"][3]) / 2)
                <= spacing * 0.45
            )
            for existing in flags
        )
        if duplicate:
            continue
        height = bbox[3] - bbox[1]
        flag_count = max(1, min(4, int(round(height / max(1.0, spacing * 1.55)))))
        flag_id = f"flag-{len(flags)}"
        geometry_confidence = min(
            1.0, 0.45 + area / max(1.0, spacing * spacing * 3.0)
        )
        if classifier_result is not None:
            confidence = 0.35 * geometry_confidence + 0.65 * classifier_result["confidence"]
        else:
            confidence = geometry_confidence
        flags.append(
            {
                "id": flag_id,
                "bbox": bbox,
                "count": flag_count,
                "direction": "down-right" if stem_up else "up-left",
                "confidence": round(float(confidence), 4),
                "source": (
                    "rhythm-cnn-fallback"
                    if inferred_from_cnn
                    else "geometry+rhythm-cnn"
                    if classifier_result is not None
                    else "stem-endpoint-geometry"
                ),
                "topology": (
                    "cnn-inferred"
                    if inferred_from_cnn
                    else "endpoint-connected"
                    if endpoint_connected
                    else "endpoint-near"
                ),
                "classifier": classifier_result,
            }
        )
        links.append({"stem_id": stem["id"], "flag_id": flag_id})
    return flags, links


def _profile_peaks(profile: np.ndarray, min_fraction: float) -> list[int]:
    """Return the center row/column of each run above a fraction of the peak."""
    if profile.size == 0:
        return []
    peak = float(profile.max())
    if peak <= 0:
        return []
    threshold = peak * min_fraction
    return [(start + end - 1) // 2 for start, end in _continuous_runs(profile >= threshold)]


def _classify_accidental_shape(glyph: np.ndarray) -> tuple[str, float]:
    """Guess sharp/flat/natural from stroke geometry alone.

    There is no trained model or labelled accidental dataset for this
    project, so this is a best-effort heuristic, not a calibrated
    classifier: it is only run on candidates that already passed the
    caller's size/position filters, and callers should treat a low
    confidence as "an accidental-shaped mark was found here" rather than
    a reliable sharp/flat/natural label.
    """
    height, width = glyph.shape
    if height == 0 or width == 0 or not np.any(glyph):
        return "unknown", 0.0

    # Sharp and natural are both built from two vertical strokes crossed by
    # two short horizontal strokes; check for that two-stroke grid first,
    # since the crossing itself can enclose a small hole that would
    # otherwise be mistaken for a flat's bowl below.  Sharp's horizontal
    # strokes overhang past the verticals on both ends; natural's stay
    # flush with them.
    column_profile = np.count_nonzero(glyph, axis=0).astype(np.float32)
    row_profile = np.count_nonzero(glyph, axis=1).astype(np.float32)
    column_peaks = _profile_peaks(column_profile, min_fraction=0.45)
    # The two verticals contribute a baseline count to every row, so the row
    # threshold must sit well above that baseline or it merges both
    # crossbar peaks into one run spanning the whole glyph height.
    row_peaks = _profile_peaks(row_profile, min_fraction=0.65)
    if len(column_peaks) >= 2 and len(row_peaks) >= 2 and height >= width * 1.15:
        left_stroke, right_stroke = column_peaks[0], column_peaks[-1]
        overhang = 0.0
        for row in row_peaks:
            run = np.flatnonzero(glyph[row] > 0)
            if run.size == 0:
                continue
            overhang = max(overhang, left_stroke - int(run.min()), int(run.max()) - right_stroke)
        if overhang > width * 0.14:
            return "sharp", 0.6
        return "natural", 0.55

    # A flat sign is a single vertical stroke with a closed bowl to its
    # right: one dominant stroke rather than the two-stroke grid above,
    # plus an enclosed hole (a contour with a parent) for the bowl.
    contours, hierarchy = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_area = 0.0
    if hierarchy is not None:
        for contour, node in zip(contours, hierarchy[0]):
            if node[3] != -1:
                hole_area = max(hole_area, float(cv2.contourArea(contour)))
    if hole_area >= width * height * 0.05 and height >= width * 1.1:
        confidence = min(0.85, 0.5 + hole_area / (width * height))
        return "flat", confidence
    return "unknown", 0.3


def detect_accidentals(
    gray: np.ndarray,
    noteheads: list[dict[str, Any]],
    stems: list[dict[str, Any]],
    spacing: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Detect sharp/flat/natural glyphs immediately left of a notehead.

    This looks only for an explicit accidental printed next to one specific
    note. It does not detect a staff's key signature, and it does not carry
    a detected accidental forward to later notes at the same pitch within a
    measure (this project does not yet detect barlines/measures reliably
    enough to model that).
    """
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    staff_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, int(round(spacing * 3.0))), 1)
    )
    long_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, staff_kernel)
    work = cv2.bitwise_and(binary, cv2.bitwise_not(long_horizontal))

    # Blank out ink that is already claimed by a notehead or a stem so a
    # note's own glyph cannot be mistaken for its neighbour's accidental.
    pad = max(1, int(round(spacing * 0.15)))
    for note in noteheads:
        x0, y0, x1, y1 = note["bbox"]
        work[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = 0
    stem_half_width = max(1, int(round(spacing * 0.18)))
    for stem in stems:
        x, top, _, bottom = stem["line"]
        work[
            max(0, top - pad) : bottom + pad,
            max(0, x - stem_half_width) : x + stem_half_width,
        ] = 0

    accidentals: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    for note in sorted(noteheads, key=lambda item: item["center"][0]):
        x0, _y0, _x1, _y1 = note["bbox"]
        note_cy = float(note["center"][1])
        search_x0 = max(0, int(round(x0 - spacing * 1.7)))
        search_x1 = max(search_x0, x0 - max(1, int(round(spacing * 0.05))))
        search_y0 = max(0, int(round(note_cy - spacing * 1.7)))
        search_y1 = min(gray.shape[0], int(round(note_cy + spacing * 1.7)))
        if search_x1 <= search_x0 or search_y1 <= search_y0:
            continue
        zone = work[search_y0:search_y1, search_x0:search_x1]
        if not np.any(zone):
            continue
        candidates: list[tuple[float, list[int]]] = []
        for lx0, ly0, lx1, ly1, area in _component_boxes((zone > 0).astype(np.uint8)):
            width = lx1 - lx0
            height = ly1 - ly0
            # A fingering/tuplet numeral sitting in this same search zone is
            # roughly one line-space tall; every accidental glyph observed
            # in spot-checks was taller than that, so this floor is set just
            # above a numeral's height rather than at the generic small-mark
            # cutoff used elsewhere in this file.
            if width < spacing * 0.12 or height < spacing * 1.05:
                continue
            if width > spacing * 1.3 or height > spacing * 3.0:
                continue
            if area < spacing * spacing * 0.05:
                continue
            bbox = [search_x0 + lx0, search_y0 + ly0, search_x0 + lx1, search_y0 + ly1]
            center_y = (bbox[1] + bbox[3]) / 2.0
            if abs(center_y - note_cy) > spacing * 1.3:
                continue
            gap = x0 - bbox[2]
            if gap < -pad or gap > spacing * 1.1:
                continue
            candidates.append((gap, bbox))
        if not candidates:
            continue
        _, bbox = min(candidates, key=lambda item: item[0])
        glyph = (work[bbox[1] : bbox[3], bbox[0] : bbox[2]] > 0).astype(np.uint8)
        accidental_type, type_confidence = _classify_accidental_shape(glyph)
        accidental_id = f"accidental-{len(accidentals)}"
        accidentals.append(
            {
                "id": accidental_id,
                "bbox": bbox,
                "type": accidental_type,
                "type_confidence": round(type_confidence, 4),
                "source": "geometry-heuristic",
            }
        )
        links.append({"notehead_id": note["id"], "accidental_id": accidental_id})
    return accidentals, links


class PrimitiveDetector:
    def __init__(
        self,
        config: DetectionConfig | None = None,
        flag_classifier: Any | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.flag_classifier = flag_classifier

    def detect(
        self,
        image: np.ndarray,
        note_score: np.ndarray,
        stem_score: np.ndarray,
        beam_score: np.ndarray | None = None,
        flag_score: np.ndarray | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if note_score.shape != image.shape[:2] or stem_score.shape != image.shape[:2]:
            raise ValueError("Segmentation maps must match the image height and width")
        if beam_score is not None and beam_score.shape != image.shape[:2]:
            raise ValueError("Beam segmentation map must match the image height and width")
        if flag_score is not None and flag_score.shape != image.shape[:2]:
            raise ValueError("Flag segmentation map must match the image height and width")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        spacing = estimate_staff_spacing(gray, self.config)
        noteheads = detect_noteheads(
            note_score, spacing, self.config.notehead_threshold
        )
        stems, notehead_stem_links = detect_stems(
            gray,
            stem_score,
            noteheads,
            spacing,
            self.config.stem_threshold,
        )
        beams, stem_beam_links = detect_beams(
            gray,
            noteheads,
            stems,
            notehead_stem_links,
            spacing,
            beam_score=beam_score,
        )
        flags, stem_flag_links = detect_flags(
            gray,
            noteheads,
            stems,
            notehead_stem_links,
            stem_beam_links,
            spacing,
            flag_classifier=self.flag_classifier,
            classifier_threshold=self.config.flag_classifier_threshold,
            fallback_classifier_threshold=self.config.flag_fallback_threshold,
            flag_score=flag_score,
            flag_threshold=self.config.flag_threshold,
        )
        accidentals, notehead_accidental_links = detect_accidentals(
            gray, noteheads, stems, spacing
        )
        return {
            "schema_version": "1.0",
            "source": source,
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "staff_spacing": round(float(spacing), 3),
            "config": asdict(self.config),
            "noteheads": noteheads,
            "stems": stems,
            "beams": beams,
            "flags": flags,
            "accidentals": accidentals,
            "relations": {
                "notehead_to_stem": notehead_stem_links,
                "stem_to_beam": stem_beam_links,
                "stem_to_flag": stem_flag_links,
                "notehead_to_accidental": notehead_accidental_links,
            },
            "counts": {
                "noteheads": len(noteheads),
                "stems": len(stems),
                "beams": len(beams),
                "flags": len(flags),
                "accidentals": len(accidentals),
            },
        }


def draw_overlay(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay = image.copy()
    for flag in result.get("flags", []):
        x0, y0, x1, y1 = flag["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (160, 40, 220), 3, cv2.LINE_AA)
    for beam in result["beams"]:
        polygon = np.asarray(beam["polygon"], dtype=np.int32)
        cv2.polylines(overlay, [polygon], True, (0, 165, 255), 3, cv2.LINE_AA)
    for stem in result["stems"]:
        x0, y0, x1, y1 = stem["line"]
        cv2.line(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2, cv2.LINE_AA)
    for note in result["noteheads"]:
        x0, y0, x1, y1 = note["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 255), 2, cv2.LINE_AA)
    for accidental in result.get("accidentals", []):
        x0, y0, x1, y1 = accidental["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), 2, cv2.LINE_AA)
    label = (
        f"noteheads={result['counts']['noteheads']}  "
        f"stems={result['counts']['stems']}  beams={result['counts']['beams']}  "
        f"flags={result['counts'].get('flags', 0)}  "
        f"accidentals={result['counts'].get('accidentals', 0)}  "
        f"staff_spacing={result['staff_spacing']}"
    )
    cv2.rectangle(overlay, (10, 10), (min(overlay.shape[1] - 10, 1180), 52), (255, 255, 255), -1)
    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return overlay


def draw_accidental_overlay(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw a diagnostic view of detected accidentals with their guessed type."""
    faded = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)
    type_colors = {
        "sharp": (40, 190, 40),
        "flat": (40, 80, 230),
        "natural": (200, 140, 0),
        "unknown": (140, 140, 140),
    }
    type_labels = {"sharp": "#", "flat": "b", "natural": "n", "unknown": "?"}
    for accidental in result.get("accidentals", []):
        x0, y0, x1, y1 = accidental["bbox"]
        acc_type = accidental.get("type", "unknown")
        color = type_colors.get(acc_type, (140, 140, 140))
        cv2.rectangle(faded, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        label = f"{type_labels.get(acc_type, '?')} {accidental.get('type_confidence', 0.0):.2f}"
        cv2.putText(
            faded, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    cv2.rectangle(faded, (10, 10), (min(faded.shape[1] - 10, 1000), 92), (255, 255, 255), -1)
    cv2.putText(faded, "GREEN #: sharp   BLUE b: flat", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        faded,
        "ORANGE n: natural   GRAY ?: shape unclear (heuristic, unverified)",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        faded,
        f"accidentals={result['counts'].get('accidentals', 0)}",
        (20, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return faded


def draw_stem_overlay(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw a high-contrast diagnostic view focused on stem quality."""
    faded = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)
    linked_noteheads = {
        relation["notehead_id"]
        for relation in result["relations"]["notehead_to_stem"]
    }
    for note in result["noteheads"]:
        x0, y0, x1, y1 = note["bbox"]
        color = (0, 215, 255) if note["id"] not in linked_noteheads else (190, 190, 190)
        cv2.rectangle(faded, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    for stem in result["stems"]:
        x0, y0, x1, y1 = stem["line"]
        color = (40, 190, 40) if stem.get("source") == "model+geometry" else (40, 80, 230)
        cv2.line(faded, (x0, y0), (x1, y1), color, 4, cv2.LINE_AA)
        cv2.circle(faded, (x0, y0), 4, color, -1, cv2.LINE_AA)
        cv2.circle(faded, (x1, y1), 4, color, -1, cv2.LINE_AA)
    cv2.rectangle(faded, (10, 10), (880, 72), (255, 255, 255), -1)
    cv2.putText(faded, "GREEN: model-supported stem", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 190, 40), 2, cv2.LINE_AA)
    cv2.putText(faded, "RED: geometry recovery   YELLOW: notehead without stem", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 80, 230), 2, cv2.LINE_AA)
    return faded


def draw_beam_overlay(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw beam candidates and the stem endpoints used to validate them."""
    faded = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)
    note_by_id = {note["id"]: note for note in result["noteheads"]}
    stem_by_id = {stem["id"]: stem for stem in result["stems"]}
    notes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for relation in result["relations"]["notehead_to_stem"]:
        note = note_by_id.get(relation["notehead_id"])
        if note is not None:
            notes_by_stem.setdefault(relation["stem_id"], []).append(note)
    for stem_id, stem in stem_by_id.items():
        linked_notes = notes_by_stem.get(stem_id, [])
        if not linked_notes:
            continue
        x, top, _, bottom = stem["line"]
        center_y = float(np.median([note["center"][1] for note in linked_notes]))
        beam_y = top if abs(center_y - top) > abs(center_y - bottom) else bottom
        cv2.circle(faded, (x, beam_y), 4, (255, 120, 0), -1, cv2.LINE_AA)
    for beam in result["beams"]:
        polygon = np.asarray(beam["polygon"], dtype=np.int32)
        color = (20, 180, 20) if beam.get("source") == "shape+stem" else (220, 80, 30)
        cv2.polylines(faded, [polygon], True, color, 4, cv2.LINE_AA)
    for flag in result.get("flags", []):
        x0, y0, x1, y1 = flag["bbox"]
        color = (0, 140, 255) if flag.get("source") == "rhythm-cnn-fallback" else (160, 40, 220)
        cv2.rectangle(faded, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    cv2.rectangle(faded, (10, 10), (980, 72), (255, 255, 255), -1)
    cv2.putText(faded, "GREEN: filled beam + stem endpoint", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 180, 20), 2, cv2.LINE_AA)
    cv2.putText(faded, "BLUE: beam recovery   PURPLE: localized flag   ORANGE: CNN inferred flag", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 80, 30), 2, cv2.LINE_AA)
    return faded


def load_legacy_segmentation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(str(path), allow_pickle=True).tolist()
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported legacy cache: {path}")
    return (
        np.asarray(payload["notehead"], dtype=np.float32),
        np.asarray(payload["stems_rests"], dtype=np.float32),
    )
