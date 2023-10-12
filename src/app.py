import cv2
import mediapipe as mp
import numpy as np

import gradio as gr
def segment(image) -> np.ndarray:
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
        return annotated_image



def main():
    webcam = gr.Image(shape=(640, 480), source="webcam", mirror_webcam=True)
    webapp = gr.interface.Interface(fn=segment, inputs=webcam, outputs="image")
    webapp.launch()


if __name__ == "__main__":
    main()
