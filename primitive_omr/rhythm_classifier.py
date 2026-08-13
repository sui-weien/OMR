from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SingleStemRhythmClassifier:
    """Validate isolated flags with the pretrained 25-omr stem CNNs."""

    class_names = ("n2", "n4", "n816")

    def __init__(self, stem_up_weights: str | Path, stem_down_weights: str | Path) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("torch is required for single-stem flag validation") from exc

        class SimpleCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(64 * 8 * 8, 512)
                self.fc2 = nn.Linear(512, 3)
                self.relu = nn.ReLU()

            def forward(self, value):
                value = self.pool(self.relu(self.conv1(value)))
                value = self.pool(self.relu(self.conv2(value)))
                value = value.reshape(-1, 64 * 8 * 8)
                value = self.relu(self.fc1(value))
                return self.fc2(value)

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[bool, Any] = {}
        for stem_up, weights in ((True, stem_up_weights), (False, stem_down_weights)):
            model = SimpleCNN().to(self.device)
            model.load_state_dict(
                torch.load(str(weights), map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models[stem_up] = model

    def _prepare(self, crop: np.ndarray):
        torch = self.torch
        height, width = crop.shape[:2]
        size = max(height, width)
        square = np.zeros((size, size), dtype=np.uint8)
        top = (size - height) // 2
        left = (size - width) // 2
        square[top : top + height, left : left + width] = crop
        resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).to(self.device)

    def validate(
        self,
        gray: np.ndarray,
        stem: dict,
        linked_notes: list[dict],
        stem_up: bool,
    ) -> dict[str, Any]:
        x, top, _, bottom = stem["line"]
        note_endpoint = bottom if stem_up else top
        note = min(
            linked_notes,
            key=lambda item: abs(item["center"][1] - note_endpoint),
        )
        x0, y0, x1, y1 = note["bbox"]
        note_width = max(1, x1 - x0)
        if stem_up:
            crop_box = (
                x0,
                max(0, top),
                min(gray.shape[1], x1 + note_width),
                min(gray.shape[0], y1),
            )
        else:
            crop_box = (
                max(0, x0 - note_width // 2),
                max(0, y0),
                min(gray.shape[1], x1),
                min(gray.shape[0], bottom),
            )
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        crop_gray = gray[crop_y0:crop_y1, crop_x0:crop_x1]
        if crop_gray.size == 0:
            return {"accepted": False, "class": "empty", "confidence": 0.0}
        _, crop = cv2.threshold(crop_gray, 200, 255, cv2.THRESH_BINARY_INV)
        tensor = self._prepare(crop)
        with self.torch.no_grad():
            logits = self.models[stem_up](tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0]
            class_index = int(self.torch.argmax(probabilities).item())
            confidence = float(probabilities[class_index].item())
        class_name = self.class_names[class_index]
        return {
            "accepted": class_name == "n816",
            "class": class_name,
            "confidence": round(confidence, 4),
            "crop_bbox": list(crop_box),
        }
