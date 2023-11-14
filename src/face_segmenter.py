import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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


# segmentation
def segment_face(image_array: np.ndarray, seg_method) -> np.ndarray:
    if seg_method == "selfie":
        mp_selfie = mp.solutions.selfie_segmentation

        with mp_selfie.SelfieSegmentation(model_selection=0) as model:
            res = model.process(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            background_mask = (res.segmentation_mask < 0.1).astype("uint8")

    elif seg_method == "face_oval":
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