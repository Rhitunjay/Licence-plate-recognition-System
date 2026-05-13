from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import cv2
import re
import numpy as np
from flask_cors import CORS
from flask import send_from_directory
import easyocr
import sqlite3
import base64
from flask import render_template

app = Flask(__name__)
CORS(app)

# MySQL connection
conn = sqlite3.connect("plates.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS plate_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    detected_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

model = YOLO(
    os.path.join(
        os.path.dirname(__file__),
        "license_plate_detector.pt"
    )
)

reader = easyocr.Reader(['en'], gpu=False)

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Run YOLO detection
    results = model(filepath)

    # Get detected bounding boxes
    detections = results[0].boxes
    
    if len(detections) == 0:
        return jsonify({"message": "Image uploaded but no license plate detected"})
    
    # Get first detected box
    box = detections[0].xyxy[0].tolist()

    x1, y1, x2, y2 = map(int, box)
    
    # Read original image using OpenCV
    image = cv2.imread(filepath)
    
    # Crop the license plate region
    cropped_plate = image[y1:y2, x1:x2]
    
    # Save cropped plate image
    cropped_path = os.path.join(UPLOAD_FOLDER, "cropped_" + file.filename)
    cv2.imwrite(cropped_path, cropped_plate)
    
    # Resize (increase size)
    cropped_plate = cv2.resize(cropped_plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Use OTSU threshold instead of adaptive
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # -------- Deskew plate safely --------
    coords = np.column_stack(np.where(thresh > 0))
    coords = coords.astype("float32")
    
    if coords.shape[0] > 0:
    
        angle = cv2.minAreaRect(coords)[-1]
    
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    
        (h, w) = cropped_plate.shape[:2]
        center = (w // 2, h // 2)
    
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        cropped_plate = cv2.warpAffine(
            cropped_plate,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
    # Recreate grayscale and threshold after rotation
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove noise and connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Sharpen characters
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    thresh = cv2.filter2D(thresh, -1, kernel)

    
   # Convert thresh image to RGB (EasyOCR expects RGB)
    rgb_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    result = reader.readtext(
        rgb_image,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False
    )

    # Sort detections by x-coordinate
    result = sorted(result, key=lambda x: (min([p[1] for p in x[0]]), min([p[0] for p in x[0]])))
    
    lines = {}

    for detection in result:
        y = int(min([p[1] for p in detection[0]]))
        text = detection[1]
    
        if y not in lines:
            lines[y] = []
    
        lines[y].append(text)
    
    plate_text = ""
    
    for key in sorted(lines.keys()):
        plate_text += "".join(lines[key])
    
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Remove country identifier
    plate_text = plate_text.replace("IND", "")
    plate_text = plate_text.replace("ND", "")
    plate_text = plate_text.replace("IN", "")
    plate_text = plate_text.replace("ID", "")

    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
    match = re.search(pattern, plate_text)
    
    if match:
        plate_text = match.group()
        
    # Save plate number to database
    if plate_text != "":
        sql = "INSERT INTO plate_history (plate_number) VALUES (%s)"
        val = (plate_text,)
        cursor.execute(sql, val)
        conn.commit()


    
    return jsonify({
        "message": "License plate detected and cropped",
        "bounding_box": box,
        "cropped_image_path": cropped_path,
        "plate_text": plate_text
    })

@app.route('/camera', methods=['POST'])
def camera_frame():

    data = request.json
    image_data = data['image']

    img_bytes = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    filepath = os.path.join(UPLOAD_FOLDER, "camera_frame.jpg")
    cv2.imwrite(filepath, frame)

    results = model(filepath)
    detections = results[0].boxes

    if len(detections) == 0:
        return jsonify({"message": "No plate detected"})

    box = detections[0].xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, box)

    cropped_plate = frame[y1:y2, x1:x2]

    cropped_path = os.path.join(UPLOAD_FOLDER, "camera_crop.jpg")
    cv2.imwrite(cropped_path, cropped_plate)

    rgb_image = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

    result = reader.readtext(
        rgb_image,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1
    )

    plate_text = ""

    for detection in result:
        plate_text += detection[1]

    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
    match = re.search(pattern, plate_text)

    if match:
        plate_text = match.group()

    if plate_text != "":
        sql = "INSERT INTO plate_history (plate_number) VALUES (%s)"
        val = (plate_text,)
        cursor.execute(sql, val)
        conn.commit()

    return jsonify({
        "plate_text": plate_text,
        "cropped_image_path": cropped_path
    })

@app.route('/history', methods=['GET'])
def get_history():

    cursor.execute("SELECT plate_number, detected_time FROM plate_history ORDER BY detected_time DESC")
    rows = cursor.fetchall()

    history = []

    for row in rows:
        history.append({
            "plate_number": row[0],
            "time": row[1].strftime("%Y-%m-%d %H:%M:%S")
        })

    return jsonify(history)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
