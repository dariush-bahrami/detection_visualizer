from dataclasses import dataclass
from typing import NamedTuple

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
class ObjectDetectionAnnotation:
    label: str
    bounding_box: BoundingBox


@dataclass
class SegmentationAnnotation:
    label: str
    mask: np.ndarray
