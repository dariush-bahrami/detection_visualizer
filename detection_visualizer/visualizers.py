import cv2
import numpy as np

from .datatypes import BoxCoordinates, Color


def visualize_bounding_box(
    image: np.ndarray,
    bounding_box: BoxCoordinates,
    label: str,
    color: Color,
    alpha: float,
) -> np.ndarray:
    font: int = cv2.FONT_HERSHEY_DUPLEX
    image_height, image_width = image.shape[:2]
    img_diagonal = (image_height**2 + image_width**2) ** 0.5
    result = image.copy()
    xmin, ymin, xmax, ymax = bounding_box
    box_diagonal = ((ymax - ymin) ** 2 + (xmax - xmin) ** 2) ** 0.5
    bbox_thickness = round((box_diagonal / img_diagonal) * img_diagonal**0.25)

    # Draw Box
    box_color = color
    cv2.rectangle(
        result, (xmin, ymin), (xmax, ymax), box_color, bbox_thickness, cv2.LINE_AA
    )
    # Draw Text

    text = label

    # Estimate initial font scale
    font_scale = 1
    font_thickness = 0
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
            xmin + bbox_thickness * 4 + text_width,
            ymin + bbox_thickness * 4 + text_height,
        ),
        [0, 0, 0],
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        text,
        (xmin + bbox_thickness, ymin + bbox_thickness * 2 + text_height),
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
    pixel = np.array(color).reshape(1, 1, 3)
    colored_image = alpha * pixel + (1 - alpha) * image
    return np.where(mask[..., None], colored_image, image).astype(image.dtype)
