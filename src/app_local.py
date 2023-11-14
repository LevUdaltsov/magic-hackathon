import random
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import PIL.ImageOps
from diffusers import StableDiffusionInpaintPipeline
from qreader import QReader

from email_utils import send_email_with_image
from face_segmenter import segment_face
from make_card import make_card
from prompts import PROMPTS, QR_MAPPING

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
    "card_image": None,
    "card_info": None,
}

# init qr reader and diffusion pipeline
qrr = QReader()

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
    image: PIL.Image.Image, contract_pixels: int
) -> Tuple[PIL.Image.Image, str]:
    supplied_prompt = "Magic kingdom tarot card, highly detailed starry sky with big full moon, ultra high resolution, artstation"

    image_array = np.asarray(image)
    qr_data = qrr.detect_and_decode(image_array)
  
    if not qr_data or len(qr_data) == 0:
        # display and error and ask user to submit another image
        gr.Warning("No card detected")
        prompt = supplied_prompt
        qr_data = "unknown"
    else:
        try:
            qr_data = qr_data[0].lower()
            qr_data = QR_MAPPING[qr_data]
        except:
            gr.Warning(f"QR code '{qr_data}' not recognized.")
            prompt = supplied_prompt
 
        prompt = random.choice(PROMPTS.get(qr_data, [supplied_prompt]))

    segmentation_mask = segment_face(image_array, seg_method=SEG_METHOD)
    segmentation_mask = contract_mask(segmentation_mask * 255, contract_pixels)

    print("===========")
    print("CARD:", qr_data)
    print("PROMPT:", prompt)
    print("===========")

    inpainted_image = pipe(
        image=image,
        mask_image=segmentation_mask,
        prompt=prompt,
        num_inference_steps=DIFFUSION_STEPS,
    ).images[0]

    try:
        card_image = make_card(inpainted_image, qr_data)
    except:
        card_image = inpainted_image

    return inpainted_image, card_image


def process_and_submit(image, email_address, submit):
    # Calculate hash of input image
    image_hash = hash(image.tobytes())

    # Process image if it has changed since last time or if cache is cleared
    if image_hash != cache["image_hash"]:
        processed_image, card_image = process_image(
            image, contract_pixels=15
        )
        cache["image_hash"] = image_hash
        cache["processed_image"] = processed_image
        cache["card_image"] = card_image
    else:
        processed_image = cache["processed_image"]
        card_image = cache["card_image"]

    # Submit email if checkbox is checked
    email_status = ""
    if submit:
        try:
            email_status = send_email_with_image(
                email_address, processed_image, card_image
            )
        except Exception as e:
            email_status = f"Error: {e}"

    return card_image, email_status


def main():
    webcam = gr.Image(
        shape=(WIDTH, HEIGHT), source="webcam", mirror_webcam=True, type="pil"
    )
    email = gr.Textbox(lines=1, placeholder="Enter your email here...", label="Email")
    submit_button = gr.Checkbox(label="Submit Email")
    webapp = gr.interface.Interface(
        fn=process_and_submit,
        inputs=[webcam, email, submit_button],
        outputs=[
            gr.Image(label="Mirror"),
            gr.Textbox(label="Email status"),
        ],
        css=css,
    )
    webapp.queue(max_size=3).launch(share=True)


if __name__ == "__main__":
    main()
