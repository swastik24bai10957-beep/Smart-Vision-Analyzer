import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess_image
from src.edges import edge_analysis
from src.features import feature_analysis
from src.segmentation import segment_image


def sample():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (170, 170), (255, 255, 255), -1)
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 3)
    return img


def test_preprocessing():
    out = preprocess_image(sample(), sigma=1.0)
    assert len(out) == 4
    assert all(img.shape == (200, 200) for img in out.values())


def test_edges():
    out = edge_analysis(sample())
    assert set(out.keys()) == {"Canny", "LoG", "DoG", "Hough", "Harris"}
    assert out["Canny"].shape == (200, 200)


def test_features():
    out = feature_analysis(sample())
    assert "HOG" in out
    assert out["HOG"].shape == (200, 200, 3)
    assert out["HOG Descriptor Length"].size == 1


def test_segmentation():
    out = segment_image(sample(), 3)
    assert out["K-Means"].shape == sample().shape
    assert out["Otsu"].shape == (200, 200)


def test_invalid_canny_thresholds():
    with pytest.raises(ValueError):
        edge_analysis(sample(), low=120, high=50)
