import cv2
import numpy as np


def preprocess_image(image, sigma=1.0):
    """Apply standard image preprocessing operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma = max(float(sigma), 0.1)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

    equalized = cv2.equalizeHist(gray)

    kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(gray, -1, kernel)

    return {
        "Grayscale": gray,
        "Gaussian Blur": blur,
        "Histogram Equalization": equalized,
        "Sharpened": sharpened,
    }
