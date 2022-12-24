from typing import NamedTuple, Optional

import numpy as np


class Color(NamedTuple):
    red: int
    green: int
    blue: int


class BoundingBox(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class ObjectDetectionGroundTruth(NamedTuple):
    label: str
    bounding_box: BoundingBox


class ObjectDetectionPrediction(NamedTuple):
    label: str
    bounding_box: BoundingBox
    confidence: float


class SegmentationGroundTruth(NamedTuple):
    label: str
    mask: np.ndarray
    bounding_box: Optional[BoundingBox] = None


class SegmentationPrediction(NamedTuple):
    label: str
    mask: np.ndarray
    confidence: float
    bounding_box: Optional[BoundingBox] = None
