# Project Statement

## Project Title

Smart Vision Analyzer

## Problem Statement

Manual inspection of images for edges, geometric structures, corners and regions can be time-consuming and inconsistent. Smart Vision Analyzer provides an interactive application that applies classical computer-vision techniques to an uploaded image and presents the outputs in a single interface.

## Scope

The project focuses on low- and mid-level computer vision for a single input image. It provides preprocessing, edge detection, feature extraction and image segmentation results. The system is intended for academic demonstration and learning.

## Target Users

- Computer Vision students
- Beginners learning OpenCV
- Faculty evaluating practical Computer Vision implementations
- Users who want to visually compare classical Computer Vision techniques

## Functional Modules

### Module 1: Image Preprocessing

- Grayscale conversion
- Gaussian filtering
- Histogram equalization
- Image sharpening

### Module 2: Edge & Feature Analysis

- Canny edge detection
- Laplacian of Gaussian
- Difference of Gaussian
- Hough line transform
- Harris corner detection
- HOG feature extraction

### Module 3: Image Segmentation

- K-Means color segmentation
- Otsu thresholding

## Non-Functional Requirements

1. **Usability:** Results are grouped into clear modules with adjustable controls.
2. **Performance:** Processing time is displayed for major analysis stages.
3. **Reliability:** Invalid image files and invalid Canny threshold relationships are handled.
4. **Maintainability:** Algorithms are separated into independent Python modules.
5. **Resource Efficiency:** Processing uses NumPy/OpenCV arrays rather than unnecessary intermediate file copies.
6. **Error Handling:** Image decoding and parameter validation are checked before processing.

## Inputs

- JPG, JPEG, PNG or BMP image
- Canny low and high thresholds
- Gaussian sigma
- K-Means cluster count

## Outputs

- Preprocessed images
- Edge maps
- Hough line visualization
- Harris corner visualization
- HOG response visualization
- K-Means segmentation
- Otsu threshold result
- Downloadable PNG results and a ZIP archive

## Academic Relevance

The project applies classical image-processing and feature-extraction techniques relevant to CSE3010 Computer Vision.
