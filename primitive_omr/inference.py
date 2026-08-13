from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def resize_for_model(image: np.ndarray, target_pixels: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    pixels = height * width
    if 3_000_000 <= pixels <= 4_350_000:
        return image, 1.0, 1.0
    ratio = (target_pixels / float(pixels)) ** 0.5
    new_width = max(32, int(round(width * ratio)))
    new_height = max(32, int(round(height * ratio)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    return resized, width / new_width, height / new_height


def _tile_positions(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def _run_tiled(
    image: np.ndarray,
    input_size: int,
    output_channels: int,
    predict: Callable[[np.ndarray], np.ndarray],
    batch_size: int = 16,
    step_size: int | None = None,
) -> np.ndarray:
    step = step_size or max(32, input_size // 2)
    original_height, original_width = image.shape[:2]
    pad_bottom = max(0, input_size - original_height)
    pad_right = max(0, input_size - original_width)
    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    height, width = padded.shape[:2]
    y_positions = _tile_positions(height, input_size, step)
    x_positions = _tile_positions(width, input_size, step)
    locations = [(x, y) for y in y_positions for x in x_positions]
    output = np.zeros((height, width, output_channels), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)

    taper = np.hanning(input_size).astype(np.float32)
    taper = np.maximum(taper, 0.12)
    blend = np.outer(taper, taper)[..., None]
    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start : start + batch_size]
        batch = np.stack(
            [padded[y : y + input_size, x : x + input_size] for x, y in batch_locations]
        )
        prediction = np.asarray(predict(batch), dtype=np.float32)
        for patch, (x, y) in zip(prediction, batch_locations):
            output[y : y + input_size, x : x + input_size] += patch * blend
            weights[y : y + input_size, x : x + input_size] += blend
    output /= np.maximum(weights, 1e-6)
    return output[:original_height, :original_width]


def run_onnx_segmentation(
    model_path: str | Path,
    image: np.ndarray,
    batch_size: int = 16,
    provider: str = "CPUExecutionProvider",
) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX inference") from exc
    session = ort.InferenceSession(str(model_path), providers=[provider])
    model_input = session.get_inputs()[0]
    input_size = int(model_input.shape[1])
    output_info = session.get_outputs()[0]
    output_channels = int(output_info.shape[-1])
    input_name = model_input.name
    output_name = output_info.name
    input_dtype = np.uint8 if "uint8" in model_input.type else np.float32

    def predict(batch: np.ndarray) -> np.ndarray:
        return session.run([output_name], {input_name: batch.astype(input_dtype, copy=False)})[0]

    return _run_tiled(image, input_size, output_channels, predict, batch_size=batch_size)


def run_keras_segmentation(
    model_path: str | Path,
    image: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required for .keras model inference") from exc
    model = tf.keras.models.load_model(str(model_path), compile=False)
    input_size = int(model.input_shape[1])
    output_channels = int(model.output_shape[-1])

    def predict(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, verbose=0)

    return _run_tiled(image, input_size, output_channels, predict, batch_size=batch_size)


def rescale_result(result: dict, scale_x: float, scale_y: float) -> dict:
    if scale_x == 1.0 and scale_y == 1.0:
        return result
    for note in result["noteheads"]:
        note["bbox"] = [
            round(note["bbox"][0] * scale_x),
            round(note["bbox"][1] * scale_y),
            round(note["bbox"][2] * scale_x),
            round(note["bbox"][3] * scale_y),
        ]
        note["center"] = [
            round(note["center"][0] * scale_x, 2),
            round(note["center"][1] * scale_y, 2),
        ]
    for stem in result["stems"]:
        x0, y0, x1, y1 = stem["line"]
        stem["line"] = [
            round(x0 * scale_x),
            round(y0 * scale_y),
            round(x1 * scale_x),
            round(y1 * scale_y),
        ]
        stem["length"] = round(stem["length"] * scale_y, 2)
    for beam in result["beams"]:
        beam["bbox"] = [
            round(beam["bbox"][0] * scale_x),
            round(beam["bbox"][1] * scale_y),
            round(beam["bbox"][2] * scale_x),
            round(beam["bbox"][3] * scale_y),
        ]
        beam["polygon"] = [
            [round(x * scale_x), round(y * scale_y)] for x, y in beam["polygon"]
        ]
        beam["length"] = round(beam["length"] * max(scale_x, scale_y), 2)
        beam["thickness"] = round(beam["thickness"] * min(scale_x, scale_y), 2)
    result["staff_spacing"] = round(result["staff_spacing"] * scale_y, 3)
    return result
