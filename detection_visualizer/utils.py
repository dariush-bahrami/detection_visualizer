from colorsys import hsv_to_rgb, rgb_to_hsv
from typing import Sequence, Union

import cv2
import numpy as np

from .datatypes import (
    BoundingBox,
    Color,
    ObjectDetectionGroundTruth,
    ObjectDetectionPrediction,
    SegmentationGroundTruth,
    SegmentationPrediction,
)


def get_mask_bbox(mask: np.ndarray):
    y, x = np.asarray(mask).nonzero()
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    return tuple(map(int, (xmin, ymin, xmax, ymax)))


def invert_color(color: Color):
    return Color(*[255 - i for i in color])


def get_adjacent_pretty_color(color: Color) -> Color:
    golden_ratio_conjugate = 0.6180339887498948
    h, s, v = rgb_to_hsv(*map(lambda x: x / 255, color))
    h += golden_ratio_conjugate
    h %= 1
    return Color(*map(lambda x: round(x * 255), hsv_to_rgb(h, s, v)))


def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def resize_image_and_annotations(
    image: np.ndarray,
    annotations: Sequence[
        Union[
            ObjectDetectionGroundTruth,
            ObjectDetectionPrediction,
            SegmentationGroundTruth,
            SegmentationPrediction,
        ]
    ],
    longest_edge_size,
    threshold=0.1,
):
    h, w = image.shape[:2]
    scale = longest_edge_size / max(h, w)
    if abs(scale - 1) < threshold:
        return image, annotations
    new_h, new_w = round(h * scale), round(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_annotations = []
    for annotation in annotations:
        if hasattr(annotation, "bounding_box"):
            # In case of segmentations, bounding_box can be None
            if annotation.bounding_box is not None:
                resized_box_coords = [i * scale for i in annotation.bounding_box]
                annotation.bounding_box = BoundingBox(*resized_box_coords)

        if hasattr(annotation, "mask"):
            annotation.mask = cv2.resize(
                annotation.mask,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
    return resized_image, resized_annotations
