import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.preprocessing import preprocess_image
from src.edges import edge_analysis
from src.features import feature_analysis
from src.segmentation import segment_image


st.set_page_config(
    page_title="Smart Vision Analyzer",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ Smart Vision Analyzer")
st.caption(
    "CSE3010 Computer Vision Project — preprocessing, edge/feature extraction, "
    "and image segmentation"
)

# ---------- Helpers ----------
def to_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png_bytes(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        save_img = img
    ok, encoded = cv2.imencode(".png", save_img)
    return encoded.tobytes() if ok else b""


def add_download(label, img, key):
    st.download_button(
        label=label,
        data=png_bytes(img),
        file_name=f"{key}.png",
        mime="image/png",
        width="stretch",
        key=f"download_{key}",
    )


def build_results_zip(groups):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for group_name, outputs in groups.items():
            for name, img in outputs.items():
                safe = name.replace("/", "_").replace(" ", "_")
                zf.writestr(f"{group_name}/{safe}.png", png_bytes(img))
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Sidebar ----------
st.sidebar.header("Analysis Controls")
low = st.sidebar.slider("Canny low threshold", 0, 255, 50)
high = st.sidebar.slider("Canny high threshold", 0, 255, 120)

if low >= high:
    st.sidebar.warning("Low threshold should normally be smaller than high threshold.")

sigma = st.sidebar.slider("Gaussian sigma", 0.5, 5.0, 1.0, 0.5)
clusters = st.sidebar.slider("K-Means clusters", 2, 6, 3)

with st.sidebar.expander("About the controls"):
    st.write(
        "Canny uses the two thresholds for hysteresis edge detection. "
        "Gaussian sigma controls smoothing strength. K-Means controls the "
        "number of color clusters used for segmentation."
    )

# ---------- Upload ----------
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG and BMP.",
)

if not uploaded:
    st.info("Upload an image to start the analysis.")

    st.markdown("## Project Modules")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. Preprocessing")
        st.write(
            "Grayscale conversion, Gaussian filtering, histogram equalization "
            "and image sharpening."
        )

    with c2:
        st.subheader("2. Edges & Features")
        st.write(
            "Canny, LoG, DoG, Hough lines, Harris corners and HOG feature analysis."
        )

    with c3:
        st.subheader("3. Segmentation")
        st.write("K-Means color segmentation and Otsu thresholding.")

    st.divider()
    st.caption(
        "Designed for academic demonstration of classical CSE3010 Computer Vision techniques."
    )
    st.stop()

# ---------- Decode ----------
data = np.frombuffer(uploaded.getvalue(), np.uint8)
image = cv2.imdecode(data, cv2.IMREAD_COLOR)

if image is None:
    st.error("The selected file could not be decoded as an image.")
    st.stop()

height, width = image.shape[:2]
pixels = height * width

st.subheader("Input Image")
st.image(to_rgb(image), width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Resolution", f"{width} × {height}")
m2.metric("Channels", str(image.shape[2] if len(image.shape) == 3 else 1))
m3.metric("Pixels", f"{pixels:,}")

st.divider()

# ---------- Processing ----------
all_results = {}

tab1, tab2, tab3 = st.tabs(
    ["1. Preprocessing", "2. Edges & Features", "3. Segmentation"]
)

with tab1:
    start = time.perf_counter()
    pre = preprocess_image(image, sigma=sigma)
    elapsed = time.perf_counter() - start
    all_results["Preprocessing"] = pre

    st.subheader("Image Preprocessing")
    st.caption(
        "Preprocessing prepares the image for later analysis by reducing noise, "
        "improving contrast and enhancing useful structures."
    )
    st.caption(f"Processing time: {elapsed * 1000:.2f} ms")

    cols = st.columns(4)
    descriptions = {
        "Grayscale": "Converts the color image to a single intensity channel.",
        "Gaussian Blur": "Smooths the image and reduces high-frequency noise.",
        "Histogram Equalization": "Improves global contrast by redistributing intensities.",
        "Sharpened": "Enhances local intensity transitions and fine details.",
    }

    for col, (name, img) in zip(cols, pre.items()):
        with col:
            st.image(to_rgb(img), caption=name, width="stretch", clamp=True)
            with st.expander("What it does"):
                st.write(descriptions[name])
            add_download("Download", img, f"preprocessing_{name.lower().replace(' ', '_')}")

with tab2:
    start = time.perf_counter()
    edge = edge_analysis(image, low, high, sigma)
    feature = feature_analysis(image)
    elapsed = time.perf_counter() - start

    combined = {**edge, "HOG": feature["HOG"]}
    all_results["Edges_Features"] = combined

    st.subheader("Edge & Feature Analysis")
    st.caption(
        "These techniques identify intensity changes, geometric structures, "
        "corners and local gradient patterns."
    )
    st.caption(f"Processing time: {elapsed * 1000:.2f} ms")

    cards = [
        ("Canny", edge["Canny"], "Multi-stage edge detector using gradient magnitude and hysteresis."),
        ("LoG", edge["LoG"], "Laplacian response after Gaussian smoothing; highlights rapid intensity changes."),
        ("DoG", edge["DoG"], "Difference between two Gaussian-smoothed images at different scales."),
        ("Hough", edge["Hough"], "Detects prominent straight line segments in the edge map."),
        ("Harris", edge["Harris"], "Detects image locations with strong changes in intensity in multiple directions."),
        ("HOG", feature["HOG"], "Visualizes local gradient orientations used by HOG to describe shape and appearance."),
    ]

    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (name, img, desc) in zip(cols, cards[row_start:row_start + 3]):
            with col:
                st.image(to_rgb(img), caption=name, width="stretch", clamp=True)
                st.caption(desc)
                add_download("Download", img, f"edges_features_{name.lower()}")

    st.info(
        f"Canny thresholds: low={low}, high={high}; Gaussian sigma={sigma:.1f}. "
        "Lower thresholds can reveal weaker edges but may also increase noise."
    )

with tab3:
    start = time.perf_counter()
    seg = segment_image(image, clusters)
    elapsed = time.perf_counter() - start
    all_results["Segmentation"] = seg

    st.subheader("Image Segmentation")
    st.caption(
        "Segmentation divides the image into meaningful intensity or color regions "
        "for simplified analysis."
    )
    st.caption(f"Processing time: {elapsed * 1000:.2f} ms")

    c1, c2 = st.columns(2)

    with c1:
        st.image(to_rgb(seg["K-Means"]), caption=f"K-Means (k={clusters})", width="stretch")
        st.caption("Groups pixels into k color clusters.")
        add_download("Download K-Means", seg["K-Means"], "segmentation_kmeans")

    with c2:
        st.image(to_rgb(seg["Otsu"]), caption="Otsu Thresholding", width="stretch")
        st.caption("Automatically selects a global threshold from the grayscale histogram.")
        add_download("Download Otsu", seg["Otsu"], "segmentation_otsu")

st.divider()
st.subheader("Export Analysis Results")

zip_data = build_results_zip(all_results)
st.download_button(
    "⬇️ Download All Results (ZIP)",
    data=zip_data,
    file_name="smart_vision_analysis_results.zip",
    mime="application/zip",
    width="stretch",
)

st.caption(
    "Tip: Use the same test image and settings when collecting screenshots for the report "
    "so your results remain consistent."
)
