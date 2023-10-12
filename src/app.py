import math
from typing import Any, List, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from sam_predict import image_to_base64, predict_custom_trained_model_sample


def segment(image) -> np.ndarray:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection_model:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        res = face_detection_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()

        best_detection = max(res.detections, key=lambda x: x.score)
        mp_drawing.draw_detection(annotated_image, best_detection)

        bounding_box = best_detection.location_data.relative_bounding_box

        return annotated_image, bounding_box


def prompt_from_qr() -> str:
    prompt = ""
    return prompt


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Tuple[int, int]:
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def segment_face(image: np.ndarray, bbox: Any) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    xmin, ymin = bbox.xmin, bbox.ymin

    xmin_px, ymin_px = _normalized_to_pixel_coordinates(xmin, ymin, image_height, image_width)
    xmax_px, ymax_px = _normalized_to_pixel_coordinates(
        xmin + bbox.width, ymin + bbox.height, image_height, image_width
    )

    bbox = np.array(xmin_px, ymin_px, xmax_px, ymax_px)

    segmentation_mask = np.array()
    # this is done on SAM
    # predictor.set_image(image)
    # masks , _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes_rescaled[None, :], multimask_output=False)
    # segmentation_mask = masks[0]
    return segmentation_mask


def inpaint_image(image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
    # this is done with stable diffusion inpainting
    return image


def process_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    annotated_image, bounding_box = detect_face(image)
    segmentation_mask = segment_face(image, bounding_box)
    # prompt = prompt_from_qr()
    # inpainted_image = inpaint_image(image, segmentation_mask, prompt)

    annotated_image = cv2.bitwise_and(image, segmentation_mask)
    return annotated_image  # replace with inpainted image


def main():
    webcam = gr.Image(shape=(640, 480), source="webcam", mirror_webcam=True)
    webapp = gr.interface.Interface(fn=process_image, inputs=webcam, outputs="image")
    webapp.launch()


if __name__ == "__main__":
    main()
