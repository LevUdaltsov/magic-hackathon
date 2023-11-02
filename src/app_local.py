import math
import random
from typing import Tuple

import cv2
import gradio as gr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from mediapipe.tasks import python as tasks

from prompts import CARD_INFO, PROMPTS, QR_MAPPING

WIDTH, HEIGHT = 512, 512

# init
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker=None,
)
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


def contract_mask(mask: np.ndarray, contract_pixels: int) -> np.ndarray:
    """Contract the mask by `contract_pixels` pixels in each direction."""
    mask = mask.copy()
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=contract_pixels)

    return mask


# segmentation
def segment_face(image: np.ndarray) -> np.ndarray:
    mp_selfie = mp.solutions.selfie_segmentation

    with mp_selfie.SelfieSegmentation(model_selection=0) as model:
        res = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        background_mask = (res.segmentation_mask < 0.1).astype("uint8")

    return background_mask


def process_image(image: np.ndarray, qr_data: str, contract_pixels: int) -> Tuple[np.ndarray, np.ndarray]:
    prompt = random.choice(PROMPTS[QR_MAPPING[qr_data]])
    card_info = CARD_INFO[QR_MAPPING[qr_data]]

    segmentation_mask = segment_face(image)

    segmentation_mask = contract_mask(segmentation_mask * 255, contract_pixels)

    print("===========")
    print("CARD:", qr_data)
    print("PROMPT:", prompt)
    print("===========")
    inpainted_image = pipe(image=image, mask_image=segmentation_mask, prompt=prompt, num_inference_steps=25).images[0]

    if card_info:
        card_info = f"## {qr_data.capitalize()}\n{card_info}"

    return inpainted_image, card_info


def main():
    webcam = gr.Image(shape=(WIDTH, HEIGHT), source="webcam", mirror_webcam=True)
    qr_datas = list(CARD_INFO.keys())
    supplied_prompt = gr.Dropdown(qr_datas, label="Card")
    contract_pixels = gr.Slider(minimum=0, maximum=50, step=1, value=15, label="Blend (pixels)")
    # webapp = gr.interface.Interface(fn=process_image, inputs=webcam, outputs="image")
    webapp = gr.interface.Interface(
        fn=process_image,
        inputs=[webcam, supplied_prompt, contract_pixels],
        outputs=[gr.Image(label="Mirror"), gr.Markdown(label="Card info")],
        css='div {margin-left: auto; margin-right: auto; width: 100%;\
            background-image: url("bg1.jpg"); repeat 0 0;}',
    )
    webapp.queue(max_size=3).launch()


if __name__ == "__main__":
    main()
