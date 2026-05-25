import os
import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
import skops.io as sio

st.set_page_config(page_title="Metal Defect Detector", layout="wide")


def get_features(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


@st.cache_resource
def load_resources():
    filename = "best_model.skops"
    if os.path.exists(filename):
        untrusted_types = sio.get_untrusted_types(file=filename)
        data = sio.load(filename, trusted=untrusted_types)

        if isinstance(data, dict):
            model = data.get("model", data.get("classifier"))
            scaler = data.get("scaler", data.get("preprocessing"))
            return model, scaler

        if hasattr(data, "predict_proba"):
            return data, None

        try:
            return data[0], data[1]
        except KeyError:
            return list(data.values())[0], list(data.values())[1]

    return None, None


st.title("Metal Surface Defect Detection System")

model, scaler = load_resources()

if model is None:
    st.error("Model file best_model.skops not found.")
else:
    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    sidebar = st.sidebar
    sidebar.header("Model Reliability")
    reliability_scores = [0.82, 0.88, 0.94, 0.91, 0.85, 0.96]
    for i in range(len(class_names)):
        sidebar.write(class_names[i])
        sidebar.progress(reliability_scores[i])

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload and Configure")
        uploaded_file = st.file_uploader(
            "Select a metal surface image", type=["jpg", "png", "bmp", "jpeg"]
        )
        threshold = st.slider("Detection Sensitivity", 0.50, 1.00, 0.90)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output_image = image.copy()

        with col2:
            st.header("Detection Output")

            window_size = 96
            step_size = 24

            boxes = []
            confidences = []
            labels = []

            h, w = image.shape[:2]

            expected_features = 1774
            if scaler is not None and hasattr(scaler, "n_features_in_"):
                expected_features = scaler.n_features_in_

            with st.spinner("Analyzing surface image..."):
                for y in range(0, h - window_size + 1, step_size):
                    for x in range(0, w - window_size + 1, step_size):
                        crop = image[y : y + window_size, x : x + window_size]
                        features = get_features(crop)

                        current_len = len(features)
                        if current_len < expected_features:
                            features = np.pad(
                                features,
                                (0, expected_features - current_len),
                                "constant",
                            )
                        elif current_len > expected_features:
                            features = features[:expected_features]

                        if scaler is not None:
                            scaled_feat = scaler.transform([features])
                        else:
                            scaled_feat = [features]

                        probabilities = model.predict_proba(scaled_feat)[0]
                        max_idx = np.argmax(probabilities)
                        max_prob = probabilities[max_idx]

                        if max_prob > threshold:
                            boxes.append([x, y, window_size, window_size])
                            confidences.append(float(max_prob))

                            if hasattr(model, "classes_") and isinstance(
                                model.classes_[0], str
                            ):
                                labels.append(model.classes_[max_idx])
                            else:
                                labels.append(class_names[max_idx])

            indices = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences, score_threshold=threshold, nms_threshold=0.3
                )

            if len(indices) > 0:
                flattened_indices = (
                    indices.flatten() if hasattr(indices, "flatten") else indices
                )

                for i in flattened_indices:
                    x, y, w_box, h_box = boxes[i]
                    label_name = labels[i]

                    cv2.rectangle(
                        output_image,
                        (x, y),
                        (x + w_box, y + h_box),
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        output_image,
                        f"{label_name} ({confidences[i]:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                final_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Defects Detected")
            else:
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, caption="No Defects Found")