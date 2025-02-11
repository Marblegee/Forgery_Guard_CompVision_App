from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ORIGINALS_FOLDER = "originals"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ORIGINALS_FOLDER"] = ORIGINALS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINALS_FOLDER, exist_ok=True)

def compare_images(original_path, uploaded_path):
    # Load images
    original = cv2.imread(original_path)
    uploaded = cv2.imread(uploaded_path)

    # Convert to grayscale
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    upload_gray = cv2.cvtColor(uploaded, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = ssim(orig_gray, upload_gray, full=True)
    diff = (diff * 255).astype("uint8")  # Normalize diff image

    # Threshold & Contours
    _, threshold = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on differences
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(uploaded, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save output images
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "diff.png"), diff)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "threshold.png"), threshold)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "original_with_diff.png"), original)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "tampered_with_diff.png"), uploaded)

    return round(score * 100, 2)  # Return similarity score

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded file
        uploaded_file = request.files["file"]
        if uploaded_file:
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(upload_path)

            # Assume there's a single original image in the folder
            original_files = os.listdir(app.config["ORIGINALS_FOLDER"])
            if not original_files:
                return "No original images found in database!"

            original_path = os.path.join(app.config["ORIGINALS_FOLDER"], original_files[0])

            # Compare images
            similarity_score = compare_images(original_path, upload_path)
            
            return render_template("results.html", score=similarity_score, image1="original_with_diff.png", image2="tampered_with_diff.png")

    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
