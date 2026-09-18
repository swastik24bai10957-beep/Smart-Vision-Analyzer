import cv2
import numpy as np


def hog_orientation_visualization(gray, cell_size=8, bins=9):
    """
    Build an interpretable HOG orientation visualization.
    Each cell shows dominant gradient orientations as line segments.
    """
    h, w = gray.shape
    h = (h // cell_size) * cell_size
    w = (w // cell_size) * cell_size
    gray = gray[:h, :w]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180.0

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bin_width = 180.0 / bins

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            mag = magnitude[y:y + cell_size, x:x + cell_size]
            ang = angle[y:y + cell_size, x:x + cell_size]

            histogram = np.zeros(bins, dtype=np.float32)

            for m, a in zip(mag.ravel(), ang.ravel()):
                b = int(a // bin_width)
                b = min(b, bins - 1)
                histogram[b] += m

            max_vote = histogram.max()
            if max_vote <= 0:
                continue

            cx = x + cell_size // 2
            cy = y + cell_size // 2

            # Draw the strongest orientations in each cell.
            for b, vote in enumerate(histogram):
                if vote < 0.35 * max_vote:
                    continue

                theta = np.deg2rad((b + 0.5) * bin_width)
                half_len = int(
                    (cell_size * 0.45) * (vote / max_vote)
                )

                dx = int(np.cos(theta) * half_len)
                dy = int(np.sin(theta) * half_len)

                cv2.line(
                    vis,
                    (cx - dx, cy - dy),
                    (cx + dx, cy + dy),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return vis


def feature_analysis(image):
    """Extract HOG features and create an orientation visualization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OpenCV HOG descriptor for actual feature extraction.
    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    resized = cv2.resize(gray, (64, 128))
    descriptor = hog.compute(resized)

    # Keep the descriptor computation explicit for the implementation,
    # while using the original image for a more useful spatial visualization.
    visual = hog_orientation_visualization(gray)

    return {
        "HOG": visual,
        "HOG Descriptor Length": np.array([len(descriptor)], dtype=np.int32),
    }
