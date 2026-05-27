from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_output_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def next_index(folder: str | Path, prefix: str, ext: str) -> int:
    """Return the next integer for files like prefix_0001.jpg."""
    folder_path = Path(folder)
    normalized_ext = ext if ext.startswith(".") else f".{ext}"
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+){re.escape(normalized_ext)}$",
        re.IGNORECASE,
    )

    max_index = -1
    if folder_path.exists():
        for file_path in folder_path.iterdir():
            match = pattern.match(file_path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 4

    height, width = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_width = text_width + padding * 2
    box_height = text_height + baseline + padding * 2

    x = max(0, min(int(x), max(0, width - box_width - 1)))
    y = max(0, min(int(y), height - 1))

    top = y - box_height
    bottom = y
    if top < 0:
        top = 0
        bottom = min(height - 1, box_height)

    cv2.rectangle(frame, (x, top), (min(width - 1, x + box_width), bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (x + padding, top + padding + text_height),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def draw_status_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a semi-transparent status panel in the top-left corner."""
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 10
    line_gap = 8

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_width = max(width for width, _ in sizes) + padding * 2
    panel_height = sum(height for _, height in sizes) + line_gap * (len(lines) - 1) + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    y = 8 + padding
    for line, (_, text_height) in zip(lines, sizes):
        y += text_height
        cv2.putText(
            frame,
            line,
            (8 + padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_gap
    return frame


def iter_image_files(source: str | Path) -> Iterable[Path]:
    source_path = Path(source)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        yield source_path
        return
    for file_path in sorted(source_path.iterdir()):
        if file_path.suffix.lower() in extensions:
            yield file_path

