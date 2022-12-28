from abc import ABC, abstractmethod
from typing import Dict, MutableMapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Color(NamedTuple):
    red: int
    green: int
    blue: int


class BoxCoordinates(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class AnnotationABC(ABC):
    @property
    @abstractmethod
    def box_label(self) -> str:
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        ...

    @property
    @abstractmethod
    def attributes(self) -> Optional[Dict]:
        ...

    @property
    @abstractmethod
    def bounding_box_coordinates(self) -> BoxCoordinates:
        ...

    @abstractmethod
    def scale(self, scale_factor: float) -> None:
        pass

    @abstractmethod
    def crop(self, crop_box_coordinates: BoxCoordinates) -> None:
        pass

    @abstractmethod
    def visualize(self, image: np.ndarray, color: Color, alpha: float) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


ColorMapper = MutableMapping[AnnotationABC, Color]


class TransformABC(ABC):
    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        annotations: Sequence[AnnotationABC],
    ) -> Tuple[np.ndarray, Sequence[AnnotationABC]]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
