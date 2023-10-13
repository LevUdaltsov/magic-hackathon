import base64

import cv2
import numpy as np
from app import *
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variables to store the captured frame and flags
captured_frame = None
capture_triggered = False


@app.route("/")
def index():
    return render_template("index.html", img_base64=captured_frame, capture_triggered=capture_triggered)


@app.route("/capture", methods=["POST"])
def capture():
    global captured_frame, capture_triggered
    ret, frame = cap.read()
    if ret:
        # Convert the captured frame to base64 for displaying in HTML
        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            captured_frame = base64.b64encode(buffer).decode("utf-8")
            capture_triggered = True
    return redirect(url_for("index"))


def custom_image_processing(image):
    # Replace this with your image processing logic
    # Here we simply invert the image
    return cv2.bitwise_not(image)


@app.route("/submit", methods=["POST"])
def submit():
    global captured_frame
    if captured_frame:
        ret, frame = cap.read()
        if ret:
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
