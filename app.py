from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the model
model = load_model('brainstroke_model.h5')

def count_and_plot_blood_clots(image_path, is_stroke):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if not is_stroke:
        return None, None  # Return None if no stroke

    # Enhance image contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Apply Gaussian Blur and Otsu's thresholding
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, brain_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to refine the mask
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # Apply mask to get brain-only region
    brain_only = cv2.bitwise_and(img, img, mask=brain_mask)

    # Detect edges using Canny
    edges = cv2.Canny(brain_only, threshold1=30, threshold2=100)
    
    # Dilate edges for better contour detection
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clot_count = 0
    min_area = 20
    max_area = 2500
    clot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if np.mean(brain_mask[y:y+h, x:x+w]) > 128:  # Ensure it's inside the brain region
                clot_count += 1
                cv2.rectangle(clot_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(clot_img, "Clot", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save processed image
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(processed_path, clot_img)

    return clot_count if clot_count > 0 else "No Clots Detected", processed_path

def classify_brain_ct(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    is_stroke = prediction > 0.5

    clot_count, processed_path = count_and_plot_blood_clots(image_path, is_stroke) if is_stroke else (None, None)
    
    return "Stroke" if is_stroke else "Normal", clot_count, processed_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_url = url_for('static', filename='uploads/' + filename)

        # Classify image and detect clots if needed
        result, clot_count, processed_path = classify_brain_ct(file_path, model)

        # If processed image exists, display it; otherwise, show original
        processed_image_url = (
            url_for('static', filename='processed/' + filename) if processed_path else file_url
        )

        return render_template("result.html", 
                               image_url=file_url, 
                               processed_image_url=processed_image_url, 
                               prediction=result, 
                               clot_count=clot_count)

    except Exception as e:
        return jsonify({"error": str(e)})

# Function to test single image manually
def test_single_image(image_path, model):
    classification, clot_count, _ = classify_brain_ct(image_path, model)
    print(f"Image: {image_path}\nClassification: {classification}\nClot Count: {clot_count}")

if __name__ == '__main__':
    app.run(debug=True)
