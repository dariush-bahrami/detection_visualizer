from dataclasses import dataclass
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


@dataclass
class ObjectDetectionGroundTruth:
    label: str
    bounding_box: BoundingBox


@dataclass
class ObjectDetectionPrediction:
    label: str
    bounding_box: BoundingBox
    confidence: float


@dataclass
class SegmentationGroundTruth:
    label: str
    mask: np.ndarray
    bounding_box: Optional[BoundingBox] = None


@dataclass
class SegmentationPrediction:
    label: str
    mask: np.ndarray
    confidence: float
    bounding_box: Optional[BoundingBox] = None
