# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from io import BytesIO

import cv2
import numpy as np
import pycocotools.mask as mask_util
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from PIL import Image

PROJECT = "241497474105"
SAM_ENDPOINT_ID = "6164623047358676992"
PIX2PIX_ENDPOINT_ID = "1192649058741649408"  # instruct pix2pix
INPAINT_ENDPOINT_ID = "8808236028625158144"  # sd-inpaint


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_str


def base64_to_image(image_str):
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    return image


def decode_rle_masks(pred_masks_rle):
    return np.stack([mask_util.decode(rle) for rle in pred_masks_rle])


def ensure_pil_image(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image


def predict(
    instances: dict,
    endpoint_id: str,
    project: str = PROJECT,
    location: str = "europe-west4",
    api_endpoint: str = "europe-west4-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in instances]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    prediction = response.predictions[0]
    return prediction


def predict_sam(
    img: Image,
):
    img = ensure_pil_image(img)
    instance = {"image": image_to_base64(img)}
    prediction = predict(instance, SAM_ENDPOINT_ID)
    prediction = dict(prediction)["masks_rle"]
    return decode_rle_masks(prediction)


def predict_pix2pix(
    img: Image,
    prompt: str,
):
    img = ensure_pil_image(img)
    instance = {
        "prompt": prompt,
        "image": image_to_base64(img),
    }
    prediction = predict(instance, PIX2PIX_ENDPOINT_ID)
    return base64_to_image(prediction)


def predict_inpaint(
    img: Image,
    mask: Image,
    prompt: str,
):
    img = ensure_pil_image(img)
    mask = ensure_pil_image(mask)
    instance = {
        "prompt": prompt,
        "image": image_to_base64(img),
        "mask_image": image_to_base64(mask),
    }
    prediction = predict(instance, INPAINT_ENDPOINT_ID)
    return base64_to_image(prediction)


# [END aiplatform_predict_custom_trained_model_sample]
