import cv2
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained face mask detection model
mask_model = load_model("mask_detector.model")

# Webcam status
webcam_active = False

def predict_mask(face_image):
    """ Predict if the face is wearing a mask """
    try:
        # Convert BGR to RGB (TensorFlow models expect RGB)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Resize to 224x224 (Ensure it matches the model input size)
        face_image = cv2.resize(face_image, (224, 224))

        # Normalize pixel values
        face_image = img_to_array(face_image) / 255.0

        # Expand dimensions to match model input
        face_image = np.expand_dims(face_image, axis=0)

        # Make prediction
        predictions = mask_model.predict(face_image)[0]

        # Ensure correct indexing for different models (binary/categorical)
        mask_prob = predictions[0]
        no_mask_prob = predictions[1]

        logging.info(f"Prediction: Mask={mask_prob:.2f}, No Mask={no_mask_prob:.2f}")  # Debugging log
        return mask_prob, no_mask_prob

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return 0, 1  # Default to "No Mask" if an error occurs

def generate_frames():
    """Generate frames for video stream"""
    global webcam_active
    cap = cv2.VideoCapture(0)  # Open the webcam
    logging.info("Webcam started...")

    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame capture failed!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(75, 75))

        if len(faces) > 0:
            logging.info(f"Detected {len(faces)} face(s) in frame.")

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            mask_prob, no_mask_prob = predict_mask(face)

            if mask_prob > no_mask_prob:
                label = f"Mask: {mask_prob * 100:.2f}%"
                color = (0, 255, 0)  # Green for mask
            else:
                label = f"No Mask: {no_mask_prob * 100:.2f}%"
                color = (0, 0, 255)  # Red for no mask

            logging.info(f"Face detected at [{x}, {y}, {w}, {h}] - {label}")

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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle the webcam detection"""
    global webcam_active
    webcam_active = not webcam_active  # Toggle webcam activity
    logging.info(f"Webcam toggled: {'Started' if webcam_active else 'Stopped'}")
    return jsonify({"status": "started" if webcam_active else "stopped"})

if __name__ == '__main__':
    #app.run(debug=True, threaded=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
