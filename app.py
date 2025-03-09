@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receive a frame from the frontend, detect mask, and return result"""
    try:
        file = request.files["frame"]
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(75, 75))

        results = []
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            mask_prob, no_mask_prob = predict_mask(face)
            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            results.append({"label": label, "mask_prob": mask_prob, "no_mask_prob": no_mask_prob})

        return jsonify({"faces_detected": len(results), "results": results})

    except Exception as e:
        return jsonify({"error": str(e)})
