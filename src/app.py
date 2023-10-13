import math
from typing import Any, List, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from api_utils import predict_inpaint, predict_pix2pix, predict_sam

WIDTH, HEIGHT = 640, 860


def pad_image(input_image):
    np.save("input_image.npy", input_image)
    # Determine the original height and width of the input image
    original_height, original_width = input_image.shape[:2]

    # Calculate the padding values needed to make the image square
    pad_width = abs(original_height - original_width) // 2

    # Create a new image with a square shape and fill with zeros
    if len(input_image.shape) == 2:
        input_image = np.expand_dims(input_image, axis=-1)
    c = input_image.shape[-1]
    if original_height > original_width:
        padded_image = np.ones((original_height, original_height, c), dtype=input_image.dtype) * 127
        padded_image[:, pad_width : pad_width + original_width, :] = input_image
    else:
        padded_image = np.ones((original_width, original_width, c), dtype=input_image.dtype) * 127
        padded_image[pad_width : pad_width + original_height, :, :] = input_image

    padded_image = np.squeeze(padded_image)

    return padded_image


def filter_masks_by_bbox(masks: np.ndarray, bbox: List) -> np.ndarray:
    """BBox has format xmin_px, ymin_px, xmax_px, ymax_px.
    Only keep masks that are completely inside bbox, and merge them.
    """
    xmin_px, ymin_px, xmax_px, ymax_px = bbox

    # check which mask fills the bbox the most
    top_mask = max(masks, key=lambda mask: mask[xmin_px:xmax_px, ymin_px:ymax_px].sum())

    return top_mask


def detect_face(image) -> np.ndarray:
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
    # prompt = "medieval fantasy painting of a wizard in a forest, artstation, ultra high resolution"
    prompt = "turn the person into a medieval fantasy painting character"
    # prompt = "give the person a mustache"
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

    bbox = [xmin_px, ymin_px, xmax_px, ymax_px]

    segmentation_masks = predict_sam(
        img=image,
    )

    segmentation_mask = filter_masks_by_bbox(segmentation_masks, bbox)

    # this is done on SAM
    # predictor.set_image(image)
    # masks , _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes_rescaled[None, :], multimask_output=False)
    # segmentation_mask = masks[0]

    # flip the mask (to inpaint the background)
    segmentation_mask = 1 - segmentation_mask

    return segmentation_mask


def inpaint_image(image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
    # this is done with stable diffusion inpainting
    # square pad image and mask
    image = pad_image(image)
    mask = pad_image(mask)
    res = predict_inpaint(image, mask * 255, prompt)
    res = np.array(res)
    return res


def process_image(image: np.ndarray, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
    annotated_image, bounding_box = detect_face(image)

    segmentation_mask = segment_face(image, bounding_box)

    # TODO
    # prompt = prompt_from_qr()

    inpainted_image = inpaint_image(image, segmentation_mask, prompt)

    return inpainted_image


def main():
    webcam = gr.Image(shape=(WIDTH, HEIGHT), source="webcam", mirror_webcam=True)
    prompt = gr.Textbox(lines=2, label="Prompt")
    # webapp = gr.interface.Interface(fn=process_image, inputs=webcam, outputs="image")
    webapp = gr.interface.Interface(fn=process_image, inputs=[webcam, prompt], outputs="image")
    webapp.launch()


if __name__ == "__main__":
    main()
