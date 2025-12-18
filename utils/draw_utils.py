import cv2
from typing import List


def draw_annotations(frame, tracks: List, count: int):
    """
    Draw bounding boxes, track ID, weight proxy, and total count.
    """

    # -------------------------------
    # Draw total count (top-left)
    # -------------------------------
    cv2.putText(
        frame,
        f"Total Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    # -------------------------------
    # Draw each tracked bird
    # -------------------------------
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Weight proxy (area-based)
        box_area = (x2 - x1) * (y2 - y1)
        weight_index = round(box_area / 1000, 2)

        label = f"ID {tr.id} | W:{weight_index}"

        # Background for text
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 5, y1),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return frame
