import cv2
import numpy as np


def _normalize(x):
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return x.astype(np.uint8)


def edge_analysis(image, low=50, high=120, sigma=1.0):
    """Run Canny, LoG, DoG, Hough and Harris analysis."""
    if low >= high:
        raise ValueError("Canny low threshold must be smaller than high threshold.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = max(float(sigma), 0.1)

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    canny = cv2.Canny(blurred, low, high)

    # Laplacian of Gaussian (LoG)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    log_img = _normalize(np.abs(lap))

    # Difference of Gaussian (DoG)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(sigma * 2.0, sigma + 0.1))
    dog = _normalize(np.abs(g1.astype(np.float32) - g2.astype(np.float32)))

    # Hough Line Transform
    lines_img = image.copy()
    lines = cv2.HoughLinesP(
        canny,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=max(20, min(image.shape[:2]) // 5),
        maxLineGap=10,
    )

    if lines is not None:
        for line in lines[:100]:
            x1, y1, x2, y2 = line.ravel()[:4]
            cv2.line(
                lines_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2,
            )

    # Harris Corner Detection
    corners_img = image.copy()
    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    if corners.size:
        threshold = 0.01 * corners.max()
        corners_img[corners > threshold] = (255, 255, 255)

    return {
        "Canny": canny,
        "LoG": log_img,
        "DoG": dog,
        "Hough": cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB),
        "Harris": cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB),
    }
