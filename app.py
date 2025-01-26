# from flask import Flask, render_template, request, jsonify
# import os
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import json

# app = Flask(__name__)

# # Load the trained model
# # MODEL_PATH = r'D:\DeepLearning Projects\Plant_Detection_Using_CNN\models\final_model.keras'
# # model = tf.keras.models.load_model(MODEL_PATH)
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/final_model.keras')
# model = tf.keras.models.load_model(MODEL_PATH)

# # Configure upload folder
# UPLOAD_FOLDER = 'static/uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Oxford 102 Flower Dataset class names
# CLASS_NAMES = [
#     'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
#     'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
#     'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
#     'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
#     'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
#     'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
#     'stemless gentian', 'artichoke', 'sweet william', 'carnation',
#     'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
#     'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
#     'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
#     'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
#     'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
#     'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
#     'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
#     'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
#     'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea',
#     'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
#     'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
#     'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
#     'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm',
#     'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
#     'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
#     'blackberry lily'
# ]

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image):
#     # Resize image to match model's expected sizing
#     img = image.resize((128, 128))  # Adjust size if your model expects different dimensions
#     # Convert to array and normalize
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = tf.expand_dims(img_array, 0)
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400

#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400

#         if file and allowed_file(file.filename):
#             # Read and preprocess the image
#             image_bytes = file.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             processed_image = preprocess_image(image)
            
#             # Make prediction
#             predictions = model.predict(processed_image)
#             predicted_class = np.argmax(predictions[0])
#             confidence = float(predictions[0][predicted_class])
            
#             # Ensure predicted_class is within bounds
#             if predicted_class >= len(CLASS_NAMES):
#                 return jsonify({'error': 'Model prediction out of bounds'}), 500
            
#             # Save the uploaded image
#             filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             image.save(filename)
            
#             return jsonify({
#                 'class': CLASS_NAMES[predicted_class],
#                 'confidence': f"{confidence:.2%}",
#                 'image_path': filename
#             })

#         return jsonify({'error': 'Invalid file type'}), 400

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/about')
# def about():
#     return render_template('about.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/final_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Oxford 102 Flower Dataset class names
CLASS_NAMES = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
    'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
    'stemless gentian', 'artichoke', 'sweet william', 'carnation',
    'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
    'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
    'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
    'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
    'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea',
    'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
    'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
    'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
    'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm',
    'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
    'blackberry lily'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Resize image to match model's expected sizing
    img = image.resize((128, 128))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Read and preprocess the image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Ensure predicted_class is within bounds
            if predicted_class >= len(CLASS_NAMES):
                return jsonify({'error': 'Model prediction out of bounds'}), 500
            
            # Save the uploaded image
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            image.save(filename)
            
            return jsonify({
                'class': CLASS_NAMES[predicted_class],
                'confidence': f"{confidence:.2%}",
                'image_path': filename
            })

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)