from typing import Sequence, Tuple

import numpy as np

from .datatypes import AnnotationABC, BoxCoordinates, TransformABC
from .utils import crop_image, get_scale_factor, scale_image


class Resize(TransformABC):
    def __init__(self, longest_edge_length: int):
        self.longest_edge_length = longest_edge_length

    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[AnnotationABC],
    ) -> Tuple[np.ndarray, Sequence[AnnotationABC]]:
        image_height, image_width = image.shape[:2]
        scale_factor = get_scale_factor(
            image_height,
            image_width,
            self.longest_edge_length,
        )
        image = scale_image(image, scale_factor)
        for annotation in annotations:
            annotation.scale(scale_factor)
        return image, annotations

    def __repr__(self):
        return f"Resize(longest_edge_length={self.longest_edge_length})"


class Crop(TransformABC):
    def __init__(self, padding: int):
        self.padding = padding

    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[AnnotationABC],
    ) -> Tuple[np.ndarray, Sequence[AnnotationABC]]:
        h, w = image.shape[:2]
        boxes = []
        for annotation in annotations:
            boxes.append(list(annotation.bounding_box_coordinates))
        boxes = np.array(boxes)

        xmin, ymin = boxes[:, 0].min(), boxes[:, 1].min()
        xmin, ymin = max(0, xmin - self.padding), max(0, ymin - self.padding)
        xmax, ymax = boxes[:, 2].max(), boxes[:, 3].max()
        xmax, ymax = min(w - 1, xmax + self.padding), min(h - 1, ymax + self.padding)

        crop_box_coordinates = BoxCoordinates(xmin, ymin, xmax, ymax)
        image = crop_image(image, crop_box_coordinates)
        for annotation in annotations:
            annotation.crop(crop_box_coordinates)
        return image, annotations

    def __repr__(self):
        return f"Crop(padding={self.padding})"


class Compose(TransformABC):
    def __init__(self, transforms: Sequence[TransformABC]):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[AnnotationABC],
    ) -> Tuple[np.ndarray, Sequence[AnnotationABC]]:
        for transform in self.transforms:
            image, annotations = transform(image, annotations)
        return image, annotations

    def __repr__(self):
        return f"Compose(transforms={self.transforms})"
