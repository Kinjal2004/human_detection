from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Set confidence threshold
conf_threshold = 0.2

@app.route('/detect_people', methods=['POST'])
def detect_people():
    # Check if the request contains an image
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    # Read the image from the request
    image_file = request.files['image']
    image_bytes = image_file.read()

    # Convert image bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Perform inference
    results = model(image)

    # Convert predictions to numpy array
    pred_numpy = results.pred[0].cpu().numpy()

    # Filter out people detections
    people_detections = pred_numpy[pred_numpy[:, 5] == 0]
    confidences = people_detections[:, 4].tolist()
    return jsonify({'confidences': confidences}), 200

if __name__ == '__main__':
    app.run(debug= True)
