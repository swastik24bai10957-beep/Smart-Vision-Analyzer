# Smart Vision Analyzer

**Student:** Swastik Pandey  
**Registration Number:** 24BAI10957  
**Institution:** VIT Bhopal University  
**Course:** CSE3010 – Computer Vision

## Overview

Smart Vision Analyzer is a modular computer-vision application that accepts an image and applies classical computer-vision techniques through an interactive Streamlit interface.

The application is organized into three major functional modules:

1. **Image Preprocessing**
   - Grayscale conversion
   - Gaussian filtering
   - Histogram equalization
   - Image sharpening

2. **Edge & Feature Analysis**
   - Canny edge detection
   - Laplacian of Gaussian (LoG)
   - Difference of Gaussian (DoG)
   - Hough line detection
   - Harris corner detection
   - Histogram of Oriented Gradients (HOG)

3. **Image Segmentation**
   - K-Means color segmentation
   - Otsu thresholding

## Main Features

- Interactive image upload
- Adjustable Canny thresholds
- Adjustable Gaussian sigma
- Adjustable K-Means cluster count
- Processing-time information
- Image-resolution statistics
- Explanations of each technique
- Individual result downloads
- Download-all-results ZIP export
- Input validation and error handling
- Pytest-based validation

## Technologies

- Python
- OpenCV
- NumPy
- Streamlit
- Pytest

## Project Structure

```text
Smart_Vision_Analyzer/
├── app.py
├── requirements.txt
├── README.md
├── statement.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── edges.py
│   ├── features.py
│   └── segmentation.py
├── tests/
│   └── test_modules.py
├── data/
└── results/
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Testing

The repository includes `pytest.ini`, so the `src` package is discovered automatically by pytest.

Run:

```bash
pytest -q
```

The test suite checks preprocessing, edge detection, HOG feature extraction, segmentation, and invalid parameter handling.

## Academic Alignment

The project demonstrates CSE3010 concepts from image preprocessing and filtering through feature extraction and image segmentation.

## Notes

OpenCV is pinned to version `4.13.0.92` because the project uses the OpenCV 4 Python API for `cv2.HOGDescriptor`.

HOG is extracted with OpenCV's HOGDescriptor and visualized using local gradient orientations over image cells.

The application is intended for academic demonstration and learning rather than safety-critical deployment.
