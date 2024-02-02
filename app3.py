# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from keras.models import load_model
# from PIL import Image
#
# # Load the CNN model
# model = load_model("C:/Users/ASUS/Coding Ninjas/model/corn_api.h5")
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return 'hello'
#
# @app.route("/predict", methods=['POST'])
# def predict():
#     print("hello")
#     # Load the image from the request
#     img_file = request.files['image']
#     print(img_file)
#     img = Image.open(img_file)
#     print(img)
#
#     # Preprocess the image
#     img_array = np.array(img.resize((128, 128)))
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#
#     # Pass the image to the CNN model
#     predictions = model.predict(img_array)
#
#     # Format the predictions as a JSON response
#     top_predictions = tf.keras.applications.resnet.decode_predictions(predictions, top=5)[0]
#     response = []
#     for pred in top_predictions:
#         response.append({
#             "label": pred[1],
#             "probability": float(pred[2])
#         })
#
#     return jsonify(response)
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)

# from flask import Flask, request
# import numpy as np
# from PIL import Image
# import tensorflow as tf
#
# app = Flask(__name__)
#
# MODEL = tf.keras.models.load_model("C:/Users/ASUS/Coding Ninjas\model\corn_api.h5")
#
# class TrainGenerator:
#     def __init__(self):
#         self.class_indices = {'class1': 0, 'class2': 1, 'class3': 2}
#
# @app.get('/')
# def index():
#     return 'hello'
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return 'No file part'
#
#     file = request.files['file']
#
#     # Open the image using PIL
#     image = Image.open(file)
#     resized_image = np.array(image.resize((128, 128), Image.ANTIALIAS))
#     image_array = resized_image / 255.0
#
#     # Make predictions using the loaded model
#     predictions = MODEL.predict(np.expand_dims(image_array, axis=0))
#
#     # Assuming train_generator is defined elsewhere in your code
#     # Adjust this part based on your specific implementation
#     class_indices = train_generator.class_indices
#     predicted_class = list(class_indices.keys())[np.argmax(predictions)]
#     print(f'The detected disease is: {predicted_class}')
#
#     return f'The detected disease is: {predicted_class}'
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

app = Flask(__name__)

# Load the trained CNN model
MODEL = tf.keras.models.load_model("C:/Users/ASUS/Coding Ninjas\model\cotton_api.h5")


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
    class_indices = {0: 'Aphids', 1: 'Army_worm', 2: 'Bacterial_blight', 3: 'Healthy', 4: 'Powdery_mildew', 5: 'Target_spot'}

    # Get the predicted class
    predicted_class = class_indices[np.argmax(predictions)]

    # Return the prediction as JSON
    result = {'predicted_class': predicted_class}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000, debug=True)

