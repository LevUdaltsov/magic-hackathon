import math
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
from mediapipe.tasks import python as tasks

HEIGHT, WIDTH = 512, 512

# init
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True, safety_checker=None)
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# blackout background (for sanity check of segmenter)
def blackout_background(image: np.ndarray, category_mask: np.ndarray) -> np.ndarray:
    condition = np.stack((category_mask,) * 3, axis=-1)
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    output_image = np.where(condition, image, bg_image)
    return output_image

# load img
image = mp.Image.create_from_file("alma_selfie.jpg")

# segmentation
base_options = tasks.BaseOptions(model_asset_path="models/selfie_segmenter.tflite")
options = tasks.vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
# Create the image segmenter
with tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
    segmentation_result = segmenter.segment(image)
    background_mask = segmentation_result.category_mask.numpy_view() < 1

    plt.imshow(blackout_background(image.numpy_view, background_mask))
    