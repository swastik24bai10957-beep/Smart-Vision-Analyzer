import cv2
import numpy as np


def segment_image(image, k=3):
    """Perform K-Means color segmentation and Otsu thresholding."""
    k = int(k)
    if k < 2:
        raise ValueError("K-Means requires at least 2 clusters.")

    data = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0,
    )

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return {
        "K-Means": cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        "Otsu": otsu,
    }
