import cv2
import numpy as np
import logging
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Detect if running on Render
IS_RENDER = "RENDER" in os.environ

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained face mask detection model
mask_model = load_model("mask_detector.h5")

# Webcam status
webcam_active = False
cap = None

def predict_mask(face_image):
    """ Predict if the face is wearing a mask """
    try:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (224, 224))
        face_image = img_to_array(face_image) / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        predictions = mask_model.predict(face_image, verbose=0)[0]
        mask_prob = predictions[0]
        no_mask_prob = predictions[1]
        logging.info(f"Prediction: Mask={mask_prob:.2f}, No Mask={no_mask_prob:.2f}")
        return mask_prob, no_mask_prob
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return 0, 1

def generate_frames():
    """Generate frames for video stream"""
    global webcam_active, cap
    if IS_RENDER:
        logging.warning("Webcam not available on Render.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return
    logging.info("Webcam started...")
    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame capture failed!")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(75, 75))
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            mask_prob, no_mask_prob = predict_mask(face)
            label = f"Mask: {mask_prob * 100:.2f}%" if mask_prob > no_mask_prob else f"No Mask: {no_mask_prob * 100:.2f}%"
            color = (0, 255, 0) if mask_prob > no_mask_prob else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y - 35), (x + w, y), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()
    logging.info("Webcam stopped.")

@app.route('/')
def index():
    """Route to render the HTML page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream video"""
    if IS_RENDER:
        return "Video streaming is not supported on Render.", 503
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle the webcam detection"""
    global webcam_active, cap
    if IS_RENDER:
        return jsonify({"error": "Webcam not available on Render."}), 503
    if not webcam_active:
        webcam_active = True
    else:
        webcam_active = False
        if cap:
            cap.release()
            logging.info("Webcam released.")
    logging.info(f"Webcam toggled: {'Started' if webcam_active else 'Stopped'}")
    return jsonify({"status": "started" if webcam_active else "stopped"})

@app.route('/request_camera_permission', methods=['GET'])
def request_camera_permission():
    """Endpoint to check and request camera permission."""
    return jsonify({"message": "Ensure you have allowed camera access in your browser settings."})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
