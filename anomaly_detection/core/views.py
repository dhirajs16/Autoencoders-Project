import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Load your pre-trained model
model = tf.keras.models.load_model('core/models/tb_anomaly_detector_model2.keras')

def load_and_preprocess_image(uploaded_file):
    # Read the image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # Resize and normalize
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

def detect_tb(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['xray_image']
        if uploaded_file:
            img = load_and_preprocess_image(uploaded_file)
            reconstructed = model.predict(img)
            reconstruction_error = np.mean(np.square(img - reconstructed))

            threshold = 0.01  # Adjust based on your findings
            is_anomaly = reconstruction_error > threshold
            result = 'Anomaly (TB detected)' if is_anomaly else 'Normal'

            return render(request, 'result.html', {'result': result, 'error': reconstruction_error})
    return render(request, 'upload.html')
