from typing import Union

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
from .utils import get_mask_bbox


def visualize_bounding_box(
    image: np.ndarray,
    bounding_box: BoundingBox,
    label: str,
    color: Color,
    alpha: float,
) -> np.ndarray:
    font: int = cv2.FONT_HERSHEY_DUPLEX
    image_height, image_width = image.shape[:2]
    diagonal = (image_height**2 + image_width**2) ** 0.5
    bbox_thickness = int(3e-3 * diagonal)
    result = image
    xmin, ymin, xmax, ymax = bounding_box

    # Draw Box
    box_color = color
    cv2.rectangle(
        result, (xmin, ymin), (xmax, ymax), box_color, bbox_thickness, cv2.LINE_AA
    )
    # Draw Text

    text = label

    font_thickness = int(2e-3 * diagonal)
    # Estimate initial font scale
    font_scale = (xmax - xmin) / image_height
    # Calculate text sizes
    text_width, text_height = cv2.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )[0]
    # Correct font scale to occupy specified proportion of box width
    text_width_over_box_width = min(max(1 - (xmax - xmin) / image_width, 0.25), 0.75)
    font_scale *= (text_width_over_box_width * (xmax - xmin)) / text_width
    text_width, text_height = cv2.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )[0]
    # Draw background rectangle of text
    cv2.rectangle(
        result,
        (xmin + bbox_thickness, ymin + bbox_thickness),
        (
            xmin + text_width + bbox_thickness * 4,
            ymin + text_height + bbox_thickness * 4,
        ),
        [0, 0, 0],
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        text,
        (xmin + bbox_thickness * 2, ymin + bbox_thickness * 2 + text_height),
        font,
        font_scale,
        box_color,
        font_thickness,
        cv2.LINE_AA,
    )
    result = cv2.addWeighted(result, alpha, image, 1 - alpha, 0)
    return result


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Color,
    alpha: float,
) -> np.ndarray:
    color = np.array(color).reshape(1, 1, 3)
    return np.where(mask[..., None], alpha * color + (1 - alpha) * image, image).astype(
        image.dtype
    )


def visualize_object_detection_ground_truth(
    image: np.ndarray,
    annotation: ObjectDetectionGroundTruth,
    color: Color,
    alpha: float,
):
    box_label = annotation.label
    result = visualize_bounding_box(
        image,
        annotation.bounding_box,
        box_label,
        color,
        alpha,
    )
    return result


def visualize_object_detection_prediction(
    image: np.ndarray,
    annotation: ObjectDetectionPrediction,
    color: Color,
    alpha: float,
):
    box_label = f"{annotation.label} - {annotation.confidence:.2f}"
    result = visualize_bounding_box(
        image,
        annotation.bounding_box,
        box_label,
        color,
        alpha,
    )
    return result


def visualize_segmentation_ground_truth(
    image: np.ndarray,
    annotation: SegmentationGroundTruth,
    color: Color,
    alpha: float,
):
    result = visualize_mask(image, annotation.mask, color, alpha)
    box_label = annotation.label
    if annotation.bounding_box is None:
        bbox = get_mask_bbox(annotation.mask)
    else:
        bbox = annotation.bounding_box

    result = visualize_bounding_box(
        result,
        bbox,
        box_label,
        color,
        alpha,
    )
    return result


def visualize_segmentation_prediction(
    image: np.ndarray,
    annotation: SegmentationPrediction,
    color: Color,
    alpha: float,
):
    result = visualize_mask(image, annotation.mask, color, alpha)
    box_label = f"{annotation.label} - {annotation.confidence:.2f}"
    if annotation.bounding_box is None:
        bbox = get_mask_bbox(annotation.mask)
    else:
        bbox = annotation.bounding_box

    result = visualize_bounding_box(
        result,
        bbox,
        box_label,
        color,
        alpha,
    )
    return result


VISUALIZATION_FUNCTIONS = {
    ObjectDetectionGroundTruth: visualize_object_detection_ground_truth,
    ObjectDetectionPrediction: visualize_object_detection_prediction,
    SegmentationGroundTruth: visualize_segmentation_ground_truth,
    SegmentationPrediction: visualize_segmentation_prediction,
}


def visualize_annotation(
    image: np.ndarray,
    annotation: Union[
        ObjectDetectionGroundTruth,
        ObjectDetectionPrediction,
        SegmentationGroundTruth,
        SegmentationPrediction,
    ],
    color: Color,
    alpha: float,
):
    visualization_function = VISUALIZATION_FUNCTIONS[type(annotation)]
    return visualization_function(
        image,
        annotation,
        color,
        alpha,
    )
