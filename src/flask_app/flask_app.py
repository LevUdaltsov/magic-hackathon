import os
import uuid

from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import base64

from src.flask_app.app import process_image

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variables to store the captured frame and flags
captured_frame = None
capture_triggered = False
IMAGE_DIR = "captured_images"
@app.route('/')
def index():
    return render_template("index.html", img_base64=captured_frame, capture_triggered=capture_triggered)


@app.route("/capture", methods=["POST"])
def capture():
    global capture_triggered, captured_frame
    ret, frame = cap.read()
    if ret:
        # Save the captured frame as a unique PNG file
        unique_id = str(uuid.uuid4())
        image_filename = os.path.join(IMAGE_DIR, f"{unique_id}.png")
        cv2.imwrite(image_filename, frame)
        # Convert the captured frame to base64 for displaying in HTML
        ret2, buffer = cv2.imencode('.jpg', frame)
        if ret2:
            captured_frame = base64.b64encode(buffer).decode('utf-8')
        capture_triggered = True
    return redirect(url_for("index"))


def custom_image_processing(image):
    # Replace this with your image processing logic
    # Here we simply invert the image
    return cv2.bitwise_not(image)


@app.route('/submit', methods=['POST'])
def submit():
    global captured_frame
    if capture_triggered:
        # Get the most recently captured image
        image_files = os.listdir(IMAGE_DIR)
        if image_files:
            latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(IMAGE_DIR, x)))

            # Load and process the image
            image_path = os.path.join(IMAGE_DIR, latest_image)
            frame = cv2.imread(image_path)
            processed_image, text = process_image(frame, contract_pixels=15)
            ret, buffer = cv2.imencode(".jpg", processed_image)
            if ret:
                captured_frame = base64.b64encode(buffer).decode("utf-8")
                return render_template("result.html", img_base64=captured_frame, text=text)
    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear():
    global captured_frame, captured
    captured_frame = None
    captured = False
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
