# Smart Vision Analyzer V3

## Improvements over V2

- Added `pytest.ini` so pytest can discover `src` without setting `PYTHONPATH`.
- Improved HOG visualization using local gradient orientations over image cells.
- Retained OpenCV HOGDescriptor for actual feature extraction.
- Added a test for the HOG descriptor length.
- Updated the README with the new HOG and test configuration.
