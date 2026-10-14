# Deepfake Image Detector

A college project for detecting whether an image is AI-generated or a real photo using deep learning, with a simple Streamlit web interface.

## Features
- Upload and test images (JPEG, PNG, WebP)
- Detects AI generated images vs real photographs
- Gives a confidence score and prediction (Likely AI, Likely Real, or Inconclusive)
- Keeps a local history of analyzed images using SQLite

## Requirements
- Python 3.10+
- PyTorch and dependencies in `requirements.txt`

## Setup & Running

1. Clone or download the repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` in your web browser.

## Project Structure
- `app.py` - main entry point for the Streamlit web app
- `pages/` - UI pages for uploading images and viewing history
- `src/` - core logic for image loading, preprocessing, model inference, and storage
- `artifacts/` - trained model weights and calibration metadata
- `configs/` - configuration files for dataset and model settings
- `tests/` - test scripts to verify model loading and functionality
