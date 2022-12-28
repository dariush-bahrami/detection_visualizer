from collections import OrderedDict
from colorsys import hsv_to_rgb, rgb_to_hsv
from typing import MutableMapping

import cv2
import numpy as np

from .datatypes import AnnotationABC, BoxCoordinates, Color


def get_scale_factor(image_height: int, image_width: int, longest_edge_size: int):
    scale = longest_edge_size / max(image_height, image_width)
    return scale


def scale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    new_h, new_w = round(h * scale_factor), round(w * scale_factor)
    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    return image


def crop_image(image: np.ndarray, crop_box_coordinates: BoxCoordinates) -> np.ndarray:
    xmin, ymin, xmax, ymax = crop_box_coordinates
    return image[ymin:ymax, xmin:xmax]


def get_mask_bbox(mask: np.ndarray) -> BoxCoordinates:
    y, x = np.asarray(mask).nonzero()
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    return BoxCoordinates(*[int(i) for i in (xmin, ymin, xmax, ymax)])


def get_adjacent_pretty_color(color: Color) -> Color:
    golden_ratio_conjugate = (5**0.5 - 1) / 2
    h, s, v = rgb_to_hsv(*map(lambda x: x / 255, color))
    h += golden_ratio_conjugate
    h %= 1
    return Color(*map(lambda x: round(x * 255), hsv_to_rgb(h, s, v)))


class AutoColorMapper(MutableMapping):
    def __init__(self, first_color: Color = Color(0, 255, 0)):
        self.first_color = first_color
        self.__collection = OrderedDict()

    def __getitem__(self, annotation: AnnotationABC) -> Color:
        if annotation.category in self.__collection:
            return self.__collection[annotation.category]
        elif len(self) == 0:
            self.__collection[annotation.category] = self.first_color
            return self.first_color
        else:
            last_category = next(reversed(self.__collection))
            last_color = self.__collection[last_category]
            new_color = get_adjacent_pretty_color(last_color)
            self.__collection[annotation.category] = new_color
            return new_color

    def __setitem__(self, annotation: AnnotationABC, color: Color):
        self.__collection[annotation.category] = color

    def __delitem__(self, annotation: AnnotationABC):
        del self.__collection[annotation.category]

    def __iter__(self):
        return iter(self.__collection)

    def __len__(self):
        return len(self.__collection)

    def __repr__(self):
        return repr(self.__collection)
