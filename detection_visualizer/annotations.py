from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from .datatypes import AnnotationABC, BoxCoordinates, Color
from .utils import crop_image, get_mask_bbox, scale_image
from .visualizers import visualize_bounding_box, visualize_mask


class BoundingBox(AnnotationABC):
    def __init__(
        self,
        box_label: str,
        category: str,
        coordinates: BoxCoordinates,
        attributes: Optional[Dict] = None,
    ):
        self.__box_label = box_label
        self.__category = category
        self.__coordinates = coordinates
        self.__attributes = attributes

    @property
    def box_label(self) -> str:
        return self.__box_label

    @property
    def category(self) -> str:
        return self.__category

    @property
    def attributes(self) -> Optional[Dict]:
        return self.__attributes

    @property
    def bounding_box_coordinates(self) -> BoxCoordinates:
        return self.__coordinates

    def scale(self, scale_factor: float) -> None:
        xmin, ymin, xmax, ymax = [
            round(i * scale_factor) for i in self.bounding_box_coordinates
        ]
        self.__coordinates = BoxCoordinates(xmin, ymin, xmax, ymax)

    def crop(self, crop_box_coordinates: BoxCoordinates) -> None:
        cbox_xmin, cbox_ymin, cbox_xmax, cbox_ymax = crop_box_coordinates
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = self.bounding_box_coordinates
        bbox_xmin -= cbox_xmin
        bbox_ymin -= cbox_ymin
        bbox_xmax -= cbox_xmin
        bbox_ymax -= cbox_ymin
        self.__coordinates = BoxCoordinates(
            bbox_xmin,
            bbox_ymin,
            bbox_xmax,
            bbox_ymax,
        )

    def visualize(self, image: np.ndarray, color: Color, alpha: float):
        image = visualize_bounding_box(
            image,
            self.bounding_box_coordinates,
            self.box_label,
            color,
            alpha,
        )
        return image

    def __repr__(self):
        return f"BoundingBox(box_label={self.box_label}, category={self.category})"


class Mask(AnnotationABC):
    def __init__(
        self,
        box_label: str,
        category: str,
        mask_array: np.ndarray,
        attributes: Optional[Dict] = None,
    ):
        self.__box_label = box_label
        self.__category = category
        self.__mask_array = mask_array
        self.__attributes = attributes

    @property
    def box_label(self) -> str:
        return self.__box_label

    @property
    def category(self) -> str:
        return self.__category

    @property
    def attributes(self) -> Optional[Dict]:
        return self.__attributes

    @property
    def mask_array(self) -> np.ndarray:
        return self.__mask_array

    @property
    def bounding_box_coordinates(self) -> BoxCoordinates:
        return get_mask_bbox(self.mask_array)

    def scale(self, scale_factor: float) -> None:
        self.__mask_array = scale_image(self.__mask_array, scale_factor)

    def crop(self, crop_box_coordinates: BoxCoordinates) -> None:
        self.__mask_array = crop_image(self.__mask_array, crop_box_coordinates)

    def visualize(self, image: np.ndarray, color: Color, alpha: float):
        image = visualize_mask(image, self.mask_array, color, alpha)
        image = visualize_bounding_box(
            image,
            self.bounding_box_coordinates,
            self.box_label,
            color,
            alpha,
        )
        return image

    def __repr__(self):
        return f"Mask(box_label={self.box_label}, category={self.category})"
