from collections import OrderedDict
from typing import Sequence, Union

import numpy as np

from .datatypes import (
    Color,
    ObjectDetectionGroundTruth,
    ObjectDetectionPrediction,
    SegmentationGroundTruth,
    SegmentationPrediction,
)
from .utils import get_adjacent_pretty_color, resize_image_and_annotations
from .visualizers import visualize_annotation


class LabeToColor:
    def __init__(self, first_default_color: Color = (26, 188, 156)):
        self.first_default_color = first_default_color
        self.__collection = OrderedDict()

    def __getitem__(self, label: str):
        if label in self.__collection:
            return self.__collection[label]
        elif len(self.__collection) == 0:
            self.__collection[label] = self.first_default_color
            return self.__collection[label]
        else:
            last_label = next(reversed(self.__collection))
            last_color = self.__collection[last_label]
            self.__collection[label] = get_adjacent_pretty_color(last_color)
            return self.__collection[label]

    def __setitem__(self, label: str, color: Color):
        self.__collection[label] = color

    def __repr__(self):
        return repr(self.__collection)

    def __str__(self):
        return str(self.__collection)


class Visualizer:
    def __init__(
        self,
        min_alpha: float = 0.8,
        max_alpha: float = 0.9,
        longest_edge_length: int = 512,
    ):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.longest_edge_length = longest_edge_length
        self.label_to_color = LabeToColor()

    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[
            Union[
                ObjectDetectionGroundTruth,
                ObjectDetectionPrediction,
                SegmentationGroundTruth,
                SegmentationPrediction,
            ]
        ],
    ):
        image, annotations = resize_image_and_annotations(
            image, annotations, self.longest_edge_length
        )
        result = np.asarray(image).copy()
        alphas = np.linspace(self.max_alpha, self.min_alpha, len(annotations)).tolist()
        for annotation in annotations:
            color = self.label_to_color[annotation.label]
            alpha = alphas.pop()
            result = visualize_annotation(result, annotation, color, alpha)
        return result