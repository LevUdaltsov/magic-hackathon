import cv2
import mediapipe as mp
import numpy as np
from pyzbar.pyzbar import decode

import requests
from PIL import Image
from io import BytesIO

import gradio as gr


def detect_and_extract_qr_code(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect QR codes in the grayscale image
    decoded_objects = decode(gray)
    if decoded_objects:
        for obj in decoded_objects:
            # Extract the QR code location and data
            points = obj.polygon
            if len(points) == 4:
                # Transform the QR code region into a rectangular shape
                rect = np.zeros((4, 2), dtype="float32")
                for j, point in enumerate(points):
                    rect[j] = [point.x, point.y]

                width, height = 200, 200  # Define the size of the output QR code image
                destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                                       dtype="float32")

                # Compute the perspective transformation matrix and apply it
                matrix = cv2.getPerspectiveTransform(rect, destination)
                warped = cv2.warpPerspective(image, matrix, (width, height))

                # Decode the QR code data
                qr_data = obj.data.decode('utf-8')
                print(f"Detected QR Code: {qr_data}")

                return warped, qr_data
    return None, None


def segment(image) -> np.ndarray:

    # Creating annotated image from face

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils


    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection_model:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        res = face_detection_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        annotated_image = image.copy()
        for detection in res.detections:

            # we need detection.location_data.relative_bounding_box to send to SAM
            mp_drawing.draw_detection(annotated_image, detection)
    qr_code_image, image_url = detect_and_extract_qr_code(image)
    print("-" * 40)
    print("IMAGE URL: ", image_url)
    response = requests.get(image_url)

    if response.status_code == 200:
        # Open the image from the response content
        image = Image.open(BytesIO(response.content))

        # Convert the PIL image to a NumPy array
        image_array = np.array(image)

    if qr_code_image is not None:
        # Save the extracted QR code as a new image
        cv2.imwrite('extracted_qr_code.jpg', qr_code_image)

    return image_array


def main():
    webcam = gr.Image(shape=(640, 480), source="webcam", mirror_webcam=True)
    webapp = gr.interface.Interface(fn=segment, inputs=webcam, outputs="image")
    webapp.launch()


if __name__ == "__main__":
    main()
