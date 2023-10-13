import cv2
import numpy as np
from pyzbar.pyzbar import decode


def detect_qr(image):
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
                destination = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
                )

                # Compute the perspective transformation matrix and apply it
                matrix = cv2.getPerspectiveTransform(rect, destination)
                warped = cv2.warpPerspective(image, matrix, (width, height))

                # Decode the QR code data
                qr_data = obj.data.decode("utf-8")
                print(f"Detected QR Code: {qr_data}")

                return qr_data
    return None
