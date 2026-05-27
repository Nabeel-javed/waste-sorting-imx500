from __future__ import annotations

import cv2
import numpy as np

from src.bin_logic import UNKNOWN_BIN, get_recommended_bin


ZONE_BINS: dict[str, set[str]] = {
    "left": {"Plastic / Packaging"},
    "center": {"Paper", "Paper / Cardboard"},
    "right": {"Glass", "Metal / Recycling", "General Waste / Packaging"},
}

ZONE_LABELS: dict[str, str] = {
    "left": "PLASTIC",
    "center": "PAPER / CARDBOARD",
    "right": "GLASS / METAL / GENERAL",
}


def get_object_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_zone_from_center(center_x: int, frame_width: int) -> str:
    first_boundary = frame_width / 3
    second_boundary = 2 * frame_width / 3
    if center_x < first_boundary:
        return "left"
    if center_x < second_boundary:
        return "center"
    return "right"


def expected_zone(class_name: str) -> str | None:
    recommended_bin = get_recommended_bin(class_name)
    if recommended_bin == UNKNOWN_BIN:
        return None
    for zone_name, accepted_bins in ZONE_BINS.items():
        if recommended_bin in accepted_bins:
            return zone_name
    return None


def check_zone_correctness(class_name: str, zone_name: str) -> bool:
    recommended_bin = get_recommended_bin(class_name)
    if recommended_bin == UNKNOWN_BIN:
        return False
    return recommended_bin in ZONE_BINS.get(zone_name, set())


def _draw_dashed_vertical_line(
    frame: np.ndarray,
    x: int,
    color: tuple[int, int, int] = (180, 180, 180),
    dash_length: int = 18,
    gap: int = 12,
) -> None:
    height = frame.shape[0]
    y = 0
    while y < height:
        cv2.line(frame, (x, y), (x, min(height, y + dash_length)), color, 2)
        y += dash_length + gap


def draw_zones(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    boundary_one = int(width / 3)
    boundary_two = int(2 * width / 3)

    overlay = frame.copy()
    zone_colors = {
        "left": (60, 120, 60),
        "center": (120, 100, 40),
        "right": (90, 70, 120),
    }
    cv2.rectangle(overlay, (0, 0), (boundary_one, height), zone_colors["left"], -1)
    cv2.rectangle(overlay, (boundary_one, 0), (boundary_two, height), zone_colors["center"], -1)
    cv2.rectangle(overlay, (boundary_two, 0), (width, height), zone_colors["right"], -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    _draw_dashed_vertical_line(frame, boundary_one)
    _draw_dashed_vertical_line(frame, boundary_two)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_positions = {
        "left": (15, 28),
        "center": (boundary_one + 15, 28),
        "right": (boundary_two + 15, 28),
    }
    for zone_name, label in ZONE_LABELS.items():
        cv2.putText(
            frame,
            label,
            label_positions[zone_name],
            font,
            0.55,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return frame

