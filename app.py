from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
import psycopg2
import os
import cv2
import numpy as np
import io
import json
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from facenet_pytorch import InceptionResnetV1

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

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load FaceNet model
embedder = InceptionResnetV1(pretrained="vggface2").eval()

def extract_embeddings(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image format")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    embeddings_list = []
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = torch.tensor(np.array(face_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            embedding = embedder(face_tensor).numpy().tolist()
        
        embeddings_list.append(embedding)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes(), embeddings_list

@app.route("/upload", methods=["POST"])
def upload():
    try:
        name = request.form.get("name")
        image_file = request.files.get("image")

        if not name or not image_file:
            return jsonify({"error": "Missing name or image"}), 400

        image_bytes = image_file.read()
        processed_image_data, embeddings = extract_embeddings(image_bytes)

        cursor.execute(
            "INSERT INTO missing_persons (name, image, faces) VALUES (%s, %s, %s) RETURNING id",
            (name, psycopg2.Binary(processed_image_data), json.dumps(embeddings)),
        )
        conn.commit()
        image_id = cursor.fetchone()[0]

        return jsonify({
            "message": "File uploaded and processed successfully",
            "faces_detected": len(embeddings),
            "image_id": image_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/image/<int:image_id>", methods=["GET"])
def get_image(image_id):
    try:
        cursor.execute("SELECT image FROM missing_persons WHERE id = %s", (image_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Image not found"}), 404

        image_data = row[0]
        return send_file(io.BytesIO(image_data), mimetype="image/jpeg")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
