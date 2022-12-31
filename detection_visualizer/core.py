from typing import Optional, Sequence

import numpy as np

from .datatypes import AnnotationABC, ColorMapper
from .transforms import TransformABC
from .utils import AutoColorMapper


class Visualizer:
    def __init__(
        self,
        alpha: float = 0.7,
        transform: Optional[TransformABC] = None,
        color_mapper: ColorMapper = AutoColorMapper(),
    ):
        self.alpha = alpha
        self.transform = transform
        self.color_mapper = color_mapper

    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[AnnotationABC],
    ):
        if self.transform is not None:
            image, annotations = self.transform(image, annotations)
        for annotation in annotations:
            color = self.color_mapper[annotation]
            image = annotation.visualize(image, color, self.alpha)
        return image
