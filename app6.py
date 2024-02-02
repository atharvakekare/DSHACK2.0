from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

app = Flask(__name__)

# Load the trained CNN model
MODEL = tf.keras.models.load_model("C:/Users/ASUS/Coding Ninjas/model/lettuce_api.h5")


# Image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.get('/')
def index():
    return 'Hello, welcome to the Plant Disease Detection API!'


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Save the uploaded file
    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make predictions using the loaded model
    predictions = MODEL.predict(img_array)

    # Assuming class indices are consistent with your training setup
    class_indices = {0: 'Bacterial', 1: 'Fungal', 2: 'Healthy'}

    # Get the predicted class
    predicted_class = class_indices[np.argmax(predictions)]

    # Return the prediction as JSON
    result = {'predicted_class': predicted_class}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)

