from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
import psycopg2
import os
import cv2
import time
import numpy as np
import io
from PIL import Image
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Database connection
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# Face detection function (Process image in memory)
def detect_faces(image_bytes):
    # Convert image bytes to NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image format")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Encode processed image back to binary
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes(), len(faces)


# Upload image & detect faces (Without saving locally)
@app.route("/upload", methods=["POST"])
def upload():
    try:
        name = request.form.get("name")
        image_file = request.files.get("image")  # Get uploaded image

        if not name or not image_file:
            return jsonify({"error": "Missing name or image"}), 400

        # Read image data as bytes
        image_bytes = image_file.read()

        # Perform face detection
        processed_image_data, faces_count = detect_faces(image_bytes)

        # Insert processed image into PostgreSQL
        cursor.execute(
            "INSERT INTO missing_persons (name, image, faces) VALUES (%s, %s, %s) RETURNING id",
            (name, psycopg2.Binary(processed_image_data), faces_count),
        )
        conn.commit()
        image_id = cursor.fetchone()[0]

        return jsonify({
            "message": "File uploaded and processed successfully",
            "faces_detected": faces_count,
            "image_id": image_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Retrieve processed image from PostgreSQL
@app.route("/image/<int:image_id>", methods=["GET"])
def get_image(image_id):
    try:
        # Fetch image from database
        cursor.execute("SELECT image FROM missing_persons WHERE id = %s", (image_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Image not found"}), 404

        image_data = row[0]  # Binary image data

        # Serve the image as a response
        return send_file(io.BytesIO(image_data), mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
