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
from qreader import QReader

from email_utils import send_email_with_image
from prompts import CARD_INFO, PROMPTS, QR_MAPPING

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
qrr = QReader()

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker=None,
    local_files_only=True,
)
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# init face oval segmenter
mp_face_mesh = mp.solutions.face_mesh

face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
p1 = df.iloc[0]["p1"]
p2 = df.iloc[0]["p2"]

routes_idx = []
for i in range(0, df.shape[0]):
    obj = df[df["p1"] == p2]
    p1 = obj["p1"].values[0]
    p2 = obj["p2"].values[0]

    route_idx = []
    route_idx.append(p1)
    route_idx.append(p2)
    routes_idx.append(route_idx)


def contract_mask(mask: np.ndarray, contract_pixels: int) -> np.ndarray:
    """Contract the mask by `contract_pixels` pixels in each direction."""
    mask = mask.copy()
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=contract_pixels)
    return mask


# segmentation
def segment_face(image_array: np.ndarray) -> np.ndarray:
    if SEG_METHOD == "selfie":
        mp_selfie = mp.solutions.selfie_segmentation

        with mp_selfie.SelfieSegmentation(model_selection=0) as model:
            res = model.process(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            background_mask = (res.segmentation_mask < 0.1).astype("uint8")

    elif SEG_METHOD == "face_oval":
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        results = face_mesh.process(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]

        routes = []
        for source_idx, target_idx in routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (
                int(image_array.shape[1] * source.x),
                int(image_array.shape[0] * source.y),
            )
            relative_target = (
                int(image_array.shape[1] * target.x),
                int(image_array.shape[0] * target.y),
            )
            routes.append(relative_source)
            routes.append(relative_target)

        background_mask = np.zeros(
            (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
        )
        cv2.fillConvexPoly(background_mask, np.array(routes), 1)
        background_mask = 1 - background_mask

    else:
        raise ValueError("Invalid segmentation method")

    return background_mask


def process_image(
    image: PIL.Image.Image, contract_pixels: int
) -> Tuple[PIL.Image.Image, str]:
    supplied_prompt = "Magic kingdom tarot card, highly detailed starry sky with big full moon, ultra high resolution, artstation"

    image_array = np.asarray(image)
    qr_data = qrr.detect_and_decode(image_array)
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
            gr.Warning(f"QR code '{qr_data}' not recognized. Using defaults")
            prompt = supplied_prompt
        card_info = CARD_INFO.get(qr_data, "Card not available")
        prompt = random.choice(PROMPTS.get(qr_data, [supplied_prompt]))

    segmentation_mask = segment_face(image_array)
    print(contract_pixels)
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

    if card_info:
        card_info = f"## {qr_data.capitalize()}\n{card_info}"

    return inpainted_image, card_info


def process_and_submit(image, email_address, submit):
    # Calculate hash of input image
    image_hash = hash(image.tobytes())

    # Process image if it has changed since last time or if cache is cleared
    if image_hash != cache["image_hash"]:
        processed_image, card_info = process_image(image, contract_pixels=15)
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
    email = gr.Textbox(lines=1, placeholder="Enter your email here...", label="Email")
    submit_button = gr.Checkbox(label="Submit Email")
    webapp = gr.interface.Interface(
        fn=process_and_submit,
        inputs=[webcam, email, submit_button],
        outputs=[
            gr.Image(label="Mirror"),
            gr.Markdown(label="Card info"),
            gr.Textbox(label="Email status"),
        ],
        css=css,
    )
    webapp.queue(max_size=3).launch()


if __name__ == "__main__":
    main()
