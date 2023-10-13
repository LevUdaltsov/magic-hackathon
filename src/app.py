import math
import random
from typing import Any, List, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from PIL import Image

# from qr import detect_qr
from qreader import QReader

from api_utils import predict_inpaint, predict_pix2pix, predict_sam
from prompts import CARD_INFO, PROMPTS, QR_MAPPING

WIDTH, HEIGHT = 640, 640

qrr = QReader()


def overlay_mirror_border(image: np.ndarray) -> np.ndarray:
    # load the mirror png
    mirror = Image.open("mirror.png")
    mirror = mirror.resize((512 + 15, 512 + 15))

    # overlay
    image = Image.fromarray(image)
    image.paste(mirror, (-8, -8), mask=mirror)

    return np.array(image)


def filter_masks_by_bbox(masks: np.ndarray, bbox: List) -> np.ndarray:
    """BBox has format xmin_px, ymin_px, xmax_px, ymax_px.
    Only keep masks that are completely inside bbox, and merge them.
    """
    xmin_px, ymin_px, xmax_px, ymax_px = bbox

    # check which mask fills the bbox the most
    top_mask = max(masks, key=lambda mask: mask[xmin_px:xmax_px, ymin_px:ymax_px].sum())

    return top_mask


def contract_mask(mask: np.ndarray, contract_pixels: int) -> np.ndarray:
    """Contract the mask by `contract_pixels` pixels in each direction."""
    mask = mask.copy()

    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=contract_pixels)

    return mask


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

    # flip the mask (to inpaint the background)
    segmentation_mask = 1 - segmentation_mask

    return segmentation_mask


def inpaint_image(image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
    res = predict_inpaint(image, mask, prompt)
    res = np.array(res)
    return res


def process_image(image: np.ndarray, supplied_prompt: str, contract_pixels: int) -> Tuple[np.ndarray, np.ndarray]:
    qr_data = qrr.detect_and_decode(image)
    card_info = None
    if not qr_data or len(qr_data) == 0:
        # display and error and ask user to submit another image
        gr.Warning("No QR code detected. Using the supplied prompt")
        prompt = supplied_prompt
    else:
        try:
            qr_data = qr_data[0].lower()
            qr_data = QR_MAPPING[qr_data]
        except:
            gr.Warning(f"QR code '{qr_data}' not recognized. Using the supplied prompt")
            prompt = supplied_prompt
        card_info = CARD_INFO.get(qr_data, "Card not available")
        prompt = random.choice(PROMPTS.get(qr_data, [supplied_prompt]))

    _, bounding_box = detect_face(image)

    segmentation_mask = segment_face(image, bounding_box)

    segmentation_mask = contract_mask(segmentation_mask * 255, contract_pixels)

    print("===========")
    print("CARD:", qr_data)
    print("PROMPT:", prompt)
    print("===========")
    inpainted_image = inpaint_image(image, segmentation_mask, prompt)

    # inpainted_image = overlay_mirror_border(inpainted_image)

    if card_info:
        card_info = f"## {qr_data.capitalize()}\n{card_info}"

    return inpainted_image, card_info


def main():
    webcam = gr.Image(shape=(WIDTH, HEIGHT), source="webcam", mirror_webcam=True)
    supplied_prompt = gr.Textbox(lines=2, label="Fallback Prompt", value=PROMPTS["scooby-do"][0])
    contract_pixels = gr.Slider(minimum=0, maximum=50, step=1, value=15, label="Blend (pixels)")
    # webapp = gr.interface.Interface(fn=process_image, inputs=webcam, outputs="image")
    webapp = gr.interface.Interface(
        fn=process_image,
        inputs=[webcam, supplied_prompt, contract_pixels],
        outputs=[gr.Image(label="Mirror"), gr.Markdown(label="Card info")],
        css='div {margin-left: auto; margin-right: auto; width: 100%;\
            background-image: url("bg1.jpg"); repeat 0 0;}',
    )
    webapp.queue(max_size=3).launch(share=True)


if __name__ == "__main__":
    main()
