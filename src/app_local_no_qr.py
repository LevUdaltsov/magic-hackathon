import random
from typing import Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps
from diffusers import StableDiffusionInpaintPipeline

from email_utils import send_email_with_image
from face_segmenter import segment_face
from prompts import CARD_INFO, PROMPTS

WIDTH, HEIGHT = 512, 512
DIFFUSION_STEPS = 25
SEG_METHOD = "face_oval"  # "face_oval" or "selfie"

css = """
.app {
    background-size: cover;
    background-image: url("https://github.com/LevUdaltsov/magic-hackathon/blob/d11f943593020ffaf1780b0d367046a6ff4704a4/src/bg1.jpg?raw=true");
    repeat 0 0;
}
"""

# Global cache for processed image and data
cache = {
    "image_hash": None,
    "processed_image": None,
    "card_info": None,
}

# init
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker=None,
    local_files_only=True,
)
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


def contract_mask(mask: np.ndarray, contract_pixels: int) -> np.ndarray:
    """Contract the mask by `contract_pixels` pixels in each direction."""
    mask = mask.copy()
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=contract_pixels)
    return mask


def process_image(
    image: np.ndarray, card_name: str, contract_pixels: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    prompt = random.choice(PROMPTS[card_name])
    card_info = CARD_INFO[card_name]

    image_array = np.asarray(image)
    segmentation_mask = segment_face(image_array, SEG_METHOD)

    segmentation_mask = contract_mask(segmentation_mask * 255, contract_pixels)

    print("===========")
    print("CARD:", card_name)
    print("PROMPT:", prompt)
    print("===========")
    inpainted_image = pipe(
        image=image,
        mask_image=segmentation_mask,
        prompt=prompt,
        num_inference_steps=DIFFUSION_STEPS,
    ).images[0]

    if card_info:
        card_info = f"## {card_name.capitalize()}\n{card_info}"

    return inpainted_image, card_info


def process_and_submit(
    image, email_address, submit, contract_pixels=15, card_name="random"
):
    if card_name == "random":
        card_names = list(CARD_INFO.keys())
        card_name = random.choice(card_names)

    # Calculate hash of input image
    image_hash = hash(image.tobytes())

    # Process image if it has changed since last time or if cache is cleared
    if image_hash != cache["image_hash"]:
        processed_image, card_info = process_image(image, card_name, contract_pixels)
        cache["image_hash"] = image_hash
        cache["processed_image"] = processed_image
        cache["card_info"] = card_info
    else:
        processed_image = cache["processed_image"]
        card_info = cache["card_info"]

    # Submit email if checkbox is checked
    email_status = ""
    if submit:
        try:
            email_status = send_email_with_image(
                email_address, processed_image, card_info
            )
        except Exception as e:
            email_status = f"Error: {e}"

    return processed_image, card_info, email_status


def main():
    webcam = gr.Image(
        shape=(WIDTH, HEIGHT), source="webcam", mirror_webcam=True, type="pil"
    )
    # supplied_prompt = gr.Dropdown(card_names, label="Card") # uncomment to select card
    # contract_pixels = gr.Slider(minimum=0, maximum=50, step=1, value=15, label="Blend (pixels)") # uncomment to select pixel blending
    email = gr.Textbox(lines=1, placeholder="Enter your email here...", label="Email")
    submit_button = gr.Checkbox(label="Submit Email")
    webapp = gr.interface.Interface(
        fn=process_and_submit,
        # inputs=[webcam, email, submit_button, contract_pixels, supplied_prompt], # uncomment for manual control
        inputs=[webcam, email, submit_button],
        outputs=[
            gr.Image(label="Mirror"),
            gr.Markdown(label="Card info"),
            gr.Textbox(label="Email status"),
        ],
        css=css,
    )
    webapp.queue(max_size=3).launch(share=True)


if __name__ == "__main__":
    main()
