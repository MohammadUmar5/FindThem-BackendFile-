from flask import Flask, request, jsonify
from dotenv import load_dotenv
import psycopg2
import os
import cv2
import numpy as np
import torch
from PIL import Image
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

def extract_embedding(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image format")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No faces detected

    # Process the first detected face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = torch.tensor(np.array(face_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        embedding = embedder(face_tensor).numpy().flatten().tolist()

    return embedding

@app.route("/match", methods=["POST"])
def match():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "Missing image"}), 400

        image_bytes = image_file.read()
        embedding = extract_embedding(image_bytes)
        print(type(embedding))  # Should be <class 'numpy.ndarray'>
        print(embedding[:5])  # Print first few values for debugging


        if embedding is None:
            return jsonify({"error": "No face detected"}), 400

        # Query database for stored embeddings
        cursor.execute("SELECT id, name FROM missing_persons ORDER BY embedding <-> %s::vector LIMIT 1;", (embedding,))
        match = cursor.fetchone()

        if match:
            return jsonify({
                "match_found": True,
                "matched_person": {
                    "id": match[0],
                    "name": match[1]
                }
            }), 200
        else:
            return jsonify({"match_found": False, "message": "No matching person found"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
