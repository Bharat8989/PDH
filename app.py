from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained model
try:
    with open("rf_parkinsons_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


# Feature extraction function
def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    feature = np.count_nonzero(edges)

    return feature


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"result": "No file selected"})

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"result": "Invalid image"})

        feature = extract_features(img)

        feature = np.array([[feature]])

        prediction = model.predict(feature)

        if prediction[0] == 0:
            result = "Healthy"
        else:
            result = "Parkinson Risk"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": "Error processing image"})


# Run server
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)