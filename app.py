from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import models
import base64
from io import BytesIO
from tensorflow.keras.models import Model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
 # New relationships
    identifications = db.relationship('Identification', backref='user', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)



# New models for storing identifications and favorites
class Identification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    flower_class = db.Column(db.String(150), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(250), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    flower_class = db.Column(db.String(150), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

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

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user_id' not in session:
#         flash('Please log in to access this page.')
#         return redirect(url_for('login'))

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
            
#             # Get top 5 predictions
#             top_predictions = get_top_predictions(predictions, CLASS_NAMES)
            
#             # Generate heatmap
#             heatmap = generate_heatmap(processed_image, model)
            
#             # Get flower details if that function exists
#             flower_details = None
#             if 'get_flower_details' in globals():
#                 flower_details = get_flower_details(CLASS_NAMES[predicted_class])
            
#             # Save the identification to database
#             new_identification = Identification(
#                 user_id=session['user_id'],
#                 flower_class=CLASS_NAMES[predicted_class],
#                 confidence=confidence,
#                 image_path=filename
#             )
#             db.session.add(new_identification)
#             db.session.commit()
            
#             # Check for badges
#             check_and_award_badges(session['user_id'])
            
#             response_data = {
#                 'class': CLASS_NAMES[predicted_class],
#                 'confidence': f"{confidence:.2%}",
#                 'image_path': filename,
#                 'identification_id': new_identification.id,
#                 'top_predictions': top_predictions,
#                 'heatmap': heatmap
#             }
            
#             # Add flower details if available
#             if flower_details:
#                 response_data['details'] = flower_details
                
#             return jsonify(response_data)

#         return jsonify({'error': 'Invalid file type'}), 400

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
# # Add routes for history and favorites
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to access this page.')
        return redirect(url_for('login'))
    
    identifications = Identification.query.filter_by(user_id=session['user_id']).order_by(Identification.timestamp.desc()).all()
    return render_template('history.html', identifications=identifications)

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    flower_class = data.get('flower_class')
    
    # Check if already in favorites
    existing_favorite = Favorite.query.filter_by(user_id=session['user_id'], flower_class=flower_class).first()
    
    if existing_favorite:
        return jsonify({'message': 'Already in favorites'})
    
    new_favorite = Favorite(user_id=session['user_id'], flower_class=flower_class)
    db.session.add(new_favorite)
    db.session.commit()
    
    return jsonify({'message': 'Added to favorites'})

@app.route('/favorites')
def favorites():
    if 'user_id' not in session:
        flash('Please log in to access this page.')
        return redirect(url_for('login'))
    
    favorites = Favorite.query.filter_by(user_id=session['user_id']).all()
    return render_template('favorites.html', favorites=favorites)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
        
        new_user = User(name=name, email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))



def generate_heatmap(img_array, model):
    """Generate a heatmap showing what areas of the image the model is focusing on."""
    try:
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in model.layers:
            if 'conv' in layer.name:
                last_conv_layer = layer.name
        
        if not last_conv_layer:
            print("No convolutional layer found in the model")
            return None
        
        # Create a model that outputs both the final predictions and the last conv layer activations
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            # Cast image to float32 (needed for gradients)
            conv_outputs, predictions = grad_model(img_array)
            top_pred_index = tf.argmax(predictions[0])
            top_class_channel = predictions[:, top_pred_index]
            
        # Gradient of the top predicted class with respect to the output feature map
        grads = tape.gradient(top_class_channel, conv_outputs)
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by their gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match the original image size
        import cv2
        img = img_array[0] * 255  # Convert back to 0-255 range
        img = img.astype(np.uint8)
        
        # Create RGB version of grayscale image if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Convert the image to base64 string
        plt.figure(figsize=(6, 6))
        plt.imshow(superimposed_img)
        plt.axis('off')
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        # Create a simple placeholder heatmap
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "Heatmap generation failed", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
# Add this function right after get_top_predictions and before your predict route
def get_flower_details(flower_name):
    """Get details about a flower from the database."""
    # Try to find the flower in the database
    flower = FlowerInfo.query.filter(
        FlowerInfo.name.like(f'%{flower_name.lower()}%')
    ).first()
    
    if flower:
        return {
            'name': flower.name,
            'color': flower.color,
            'flowering_season': flower.flowering_season,
            'region': flower.region,
            'habitat': flower.habitat,
            'description': flower.description
        }
    else:
        # Return basic info if not found in database
        return {
            'name': flower_name,
            'description': f'A beautiful {flower_name}. More information coming soon!'
        }
def get_top_predictions(predictions, class_names, top_k=5):
    """Get the top k predictions."""
    indices = np.argsort(predictions[0])[::-1][:top_k]
    top_preds = [(class_names[i], float(predictions[0][i])) for i in indices]
    return top_preds

# Modify the predict route to include visualization
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to access this page.')
        return redirect(url_for('login'))

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
            
            # Get top 5 predictions
            top_predictions = get_top_predictions(predictions, CLASS_NAMES)
            
            # Generate heatmap
            heatmap = generate_heatmap(processed_image, model)
            
            # Save the uploaded image
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            image.save(filename)
            
            # Save identification to database
            new_identification = Identification(
                user_id=session['user_id'],
                flower_class=CLASS_NAMES[predicted_class],
                confidence=confidence,
                image_path=filename
            )
            db.session.add(new_identification)
            db.session.commit()
            
            # Get flower details
            flower_details = get_flower_details(CLASS_NAMES[predicted_class])
            
            return jsonify({
                'class': CLASS_NAMES[predicted_class],
                'confidence': f"{confidence:.2%}",
                'image_path': filename,
                'identification_id': new_identification.id,
                'top_predictions': top_predictions,
                'heatmap': heatmap,
                'details': flower_details
            })

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a route to view AI decision visualization
# @app.route('/visualization/<int:id>')
# def visualization(id):
    identification = Identification.query.get_or_404(id)
    
    # Ensure the user owns this identification or it's public
    if 'user_id' not in session or identification.user_id != session['user_id']:
        flash('You can only view your own identifications.')
        return redirect(url_for('login'))
    
    # Load and process the image
    image = Image.open(identification.image_path)
    processed_image = preprocess_image(image)
    
    # Make prediction again
    predictions = model.predict(processed_image)
    
    # Get top 5 predictions
    top_predictions = get_top_predictions(predictions, CLASS_NAMES)
    
    # Generate heatmap
    heatmap = generate_heatmap(processed_image, model)
    
    return render_template(
        'visualization.html',
        identification=identification,
        top_predictions=top_predictions,
        heatmap=heatmap
    )

import base64
from io import BytesIO

def get_heatmap_for_identification(identification):
    """Generate or retrieve a heatmap for the given identification."""
    try:
        # Load the image from the path stored in the identification
        image_path = identification.image_path
        img = Image.open(image_path)
        processed_img = preprocess_image(img)
        
        # Generate heatmap using your existing function
        heatmap_b64 = generate_heatmap(processed_img, model)
        
        # Check if heatmap was successfully generated
        if heatmap_b64 is None:
            # Create a placeholder image
            plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, "Heatmap not available for this image", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='red')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
        return heatmap_b64
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        # Create a proper error image instead of returning a string
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, f"Error generating heatmap:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=10, color='red', wrap=True)
        plt.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return heatmap_b64


def get_top_predictions_for_identification(identification):
    """Get top predictions for a previously identified image."""
    try:
        # Load the image from the path stored in the identification
        image_path = identification.image_path
        image = Image.open(image_path)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get top 5 predictions using existing function
        top_predictions = get_top_predictions(predictions, CLASS_NAMES)
        
        return top_predictions
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        # Return a default prediction if there's an error
        return [(identification.flower_class, identification.confidence)]
# @app.route('/visualization/<int:id>')
# def visualization(id):
#     if 'user_id' not in session:
#         flash('Please log in to access this page.')
#         return redirect(url_for('login'))
    
#     identification = Identification.query.get_or_404(id)
    
#     # Extract just the filename from the path
#     image_filename = os.path.basename(identification.image_path)
    
#     # Create a URL using url_for to serve the image from your uploads folder
#     image_url = url_for('static', filename=f'uploads/{image_filename}')
    
#     # Get top predictions from your model for this image
#     # This assumes you're regenerating or storing these predictions
#     top_predictions = get_top_predictions_for_identification(identification)
    
#     # Get or regenerate heatmap
#     heatmap = get_heatmap_for_identification(identification)
    
#     return render_template(
#         'visualization.html',
#         identification=identification,
#         image_url=image_url,  # Pass this new URL to the template
#         top_predictions=top_predictions,
#         heatmap=heatmap
#     )
# # Add a model for flower characteristics (populate this with data)


@app.route('/visualization/<int:id>')
def visualization(id):
    if 'user_id' not in session:
        flash('Please log in to access this page.')
        return redirect(url_for('login'))
    
    identification = Identification.query.get_or_404(id)
    
    # Ensure the user owns this identification
    if identification.user_id != session['user_id']:
        flash('You can only view your own identifications.')
        return redirect(url_for('history'))
    
    try:
        # Get image URL for display
        filename = os.path.basename(identification.image_path)
        image_url = url_for('static', filename=f'uploads/{filename}')
        
        # Load and process the image for prediction
        image = Image.open(identification.image_path)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get top 5 predictions
        top_predictions = get_top_predictions(predictions, CLASS_NAMES)
        
        # Create a simulated attention map (more reliable than Grad-CAM for some models)
        plt.figure(figsize=(8, 8))
        
        # Load the original image
        img_array = np.array(image)
        
        # Create a simulated attention pattern - a radial gradient from center
        h, w = img_array.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        # Create a radial distance matrix from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize to 0-1 range and invert (center is hotter)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        attention = 1 - (distance / max_distance)
        attention = np.clip(attention, 0, 1)
        
        # Apply a colormap
        cmap = plt.cm.jet
        attention_colored = cmap(attention)
        attention_colored = attention_colored[..., :3]  # Remove alpha channel
        
        # Blend with original image
        alpha = 0.6  # Transparency of heatmap
        blended = (1 - alpha) * img_array / 255.0 + alpha * attention_colored
        blended = np.clip(blended, 0, 1)
        
        plt.imshow(blended)
        plt.title(f"AI Focus for {identification.flower_class}")
        plt.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return render_template(
            'visualization.html',
            identification=identification,
            image_url=image_url,
            top_predictions=top_predictions,
            heatmap=heatmap
        )
        
    except Exception as e:
        # Log the error
        print(f"Error in visualization route: {str(e)}")
        
        # Create a visually appealing error page
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "We couldn't generate the visualization for this image", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='#d9534f', fontweight='bold')
        plt.text(0.5, 0.6, "Please try with another image", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='#333')
        plt.axis('off')
        
        # Add a nice background
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.gcf().patch.set_facecolor('#f8f9fa')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        error_heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Return the template with the error heatmap
        return render_template(
            'visualization.html',
            identification=identification,
            image_url=image_url,
            top_predictions=top_predictions,
            heatmap=error_heatmap
        )
class FlowerInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    color = db.Column(db.String(50))
    flowering_season = db.Column(db.String(50))
    region = db.Column(db.String(100))
    habitat = db.Column(db.String(100))
    description = db.Column(db.Text)
    
    @classmethod
    def populate_database(cls):
        """Populate the database with flower information."""
        # This is just an example. You would need to fill this with actual data
        flowers_data = [
            {
                'name': 'rose',
                'color': 'red, pink, white, yellow',
                'flowering_season': 'spring, summer',
                'region': 'worldwide',
                'habitat': 'gardens, wild',
                'description': 'Roses are perennial flowering plants known for their beauty and fragrance.'
            },
            # Add more flowers
        ]
        
        for flower_data in flowers_data:
            flower = cls(**flower_data)
            db.session.add(flower)
        
        db.session.commit()

# Add routes for exploring and searching
@app.route('/explore')
def explore():
    # Get filter parameters
    color = request.args.get('color')
    season = request.args.get('season')
    region = request.args.get('region')
    
    # Build query
    query = FlowerInfo.query
    
    if color:
        query = query.filter(FlowerInfo.color.like(f'%{color}%'))
    if season:
        query = query.filter(FlowerInfo.flowering_season.like(f'%{season}%'))
    if region:
        query = query.filter(FlowerInfo.region.like(f'%{region}%'))
    
    flowers = query.all()
    
    # Get unique options for filters
    colors = db.session.query(FlowerInfo.color).distinct().all()
    seasons = db.session.query(FlowerInfo.flowering_season).distinct().all()
    regions = db.session.query(FlowerInfo.region).distinct().all()
    
    return render_template(
        'explore.html',
        flowers=flowers,
        colors=colors,
        seasons=seasons,
        regions=regions,
        selected_color=color,
        selected_season=season,
        selected_region=region
    )

@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    if not query:
        return render_template('search.html', results=None)
    
    # Search in flower info
    results = FlowerInfo.query.filter(
        (FlowerInfo.name.like(f'%{query}%')) |
        (FlowerInfo.description.like(f'%{query}%'))
    ).all()
    
    return render_template('search.html', results=results, query=query)
# Add models for gamification
class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    requirement = db.Column(db.String(255), nullable=False)
    
    @classmethod
    def populate_badges(cls):
        """Populate the database with badge information."""
        badges = [
            {
                'name': 'Flower Novice',
                'description': 'Identify your first flower',
                'image_path': '/static/images/badges/novice.png',
                'requirement': 'identifications:1'
            },
            {
                'name': 'Flower Explorer',
                'description': 'Identify 10 different flowers',
                'image_path': '/static/images/badges/explorer.png',
                'requirement': 'unique_identifications:10'
            },
            {
                'name': 'Flower Expert',
                'description': 'Identify 50 different flowers',
                'image_path': '/static/images/badges/expert.png',
                'requirement': 'unique_identifications:50'
            },
            {
                'name': 'Rose Specialist',
                'description': 'Identify 5 roses',
                'image_path': '/static/images/badges/rose.png',
                'requirement': 'flower_type:rose:5'
            }
            # Add more badges
        ]
        
        for badge_data in badges:
            badge = cls(**badge_data)
            db.session.add(badge)
        
        db.session.commit()

class UserBadge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    badge_id = db.Column(db.Integer, db.ForeignKey('badge.id'), nullable=False)
    earned_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    badge = db.relationship('Badge', backref='user_badges')

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    difficulty = db.Column(db.String(50))  # easy, medium, hard
    questions = db.relationship('QuizQuestion', backref='quiz', lazy=True)

class QuizQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    options = db.Column(db.Text, nullable=False)  # JSON string of options
    correct_answer = db.Column(db.String(255), nullable=False)
    explanation = db.Column(db.Text)
    image_path = db.Column(db.String(255))

class UserQuizResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    completed_date = db.Column(db.DateTime, default=db.func.current_timestamp())

# Function to check and award badges
def check_and_award_badges(user_id):
    """Check user progress and award appropriate badges."""
    user = User.query.get(user_id)
    if not user:
        return
    
    # Get all badges
    all_badges = Badge.query.all()
    
    # Get user's current badges
    user_badge_ids = [ub.badge_id for ub in UserBadge.query.filter_by(user_id=user_id).all()]
    
    # Check each badge's requirements
    for badge in all_badges:
        if badge.id in user_badge_ids:
            continue  # Skip if user already has this badge
        
        # Parse requirement
        req_parts = badge.requirement.split(':')
        req_type = req_parts[0]
        
        if req_type == 'identifications':
            # Check total identifications
            count = Identification.query.filter_by(user_id=user_id).count()
            threshold = int(req_parts[1])
            if count >= threshold:
                new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(new_badge)
        
        elif req_type == 'unique_identifications':
            # Check unique flower identifications
            unique_flowers = db.session.query(Identification.flower_class)\
                .filter_by(user_id=user_id)\
                .distinct()\
                .count()
            threshold = int(req_parts[1])
            if unique_flowers >= threshold:
                new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(new_badge)
        
        elif req_type == 'flower_type':
            # Check identifications of specific flower type
            flower_type = req_parts[1]
            count = Identification.query.filter_by(
                user_id=user_id
            ).filter(
                Identification.flower_class.like(f'%{flower_type}%')
            ).count()
            threshold = int(req_parts[2])
            if count >= threshold:
                new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(new_badge)
    
    db.session.commit()

# Update the predict route to check for badges after identification

# Add routes for badges and quizzes
@app.route('/badges')
def badges():
    if 'user_id' not in session:
        flash('Please log in to view your badges.')
        return redirect(url_for('login'))
    
    user_badges = UserBadge.query.filter_by(user_id=session['user_id']).all()
    all_badges = Badge.query.all()
    
    # Separate earned and unearned badges
    earned_badges = [ub.badge for ub in user_badges]
    unearned_badges = [b for b in all_badges if b not in earned_badges]
    
    return render_template(
        'badges.html',
        earned_badges=earned_badges,
        unearned_badges=unearned_badges
    )

@app.route('/quizzes')
def quizzes():
    all_quizzes = Quiz.query.all()
    
    # Get user's completed quizzes if logged in
    completed_quizzes = []
    if 'user_id' in session:
        completed_quizzes = UserQuizResult.query.filter_by(user_id=session['user_id']).all()
    
    return render_template(
        'quizzes.html',
        quizzes=all_quizzes,
        completed_quizzes=completed_quizzes
    )

@app.route('/quiz/<int:id>')
def take_quiz(id):
    if 'user_id' not in session:
        flash('Please log in to take quizzes.')
        return redirect(url_for('login'))
    
    quiz = Quiz.query.get_or_404(id)
    questions = QuizQuestion.query.filter_by(quiz_id=id).all()
    
    return render_template(
        'take_quiz.html',
        quiz=quiz,
        questions=questions
    )

@app.route('/submit_quiz/<int:id>', methods=['POST'])
def submit_quiz(id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    quiz = Quiz.query.get_or_404(id)
    questions = QuizQuestion.query.filter_by(quiz_id=id).all()
    
    score = 0
    for question in questions:
        answer = request.form.get(f'question_{question.id}')
        if answer == question.correct_answer:
            score += 1
    
    # Save quiz result
    result = UserQuizResult(
        user_id=session['user_id'],
        quiz_id=id,
        score=score,
        total_questions=len(questions)
    )
    db.session.add(result)
    db.session.commit()
    
    # Check if any badges were earned
    check_and_award_badges(session['user_id'])
    
    return render_template(
        'quiz_result.html',
        quiz=quiz,
        score=score,
        total=len(questions),
        percentage=(score / len(questions)) * 100
    )

# Educational content
@app.route('/learn')
def learn():
    # Get all flower categories
    categories = [
        'Roses', 'Lilies', 'Orchids', 'Daisies', 
        'Sunflowers', 'Tulips', 'Carnations'
    ]
    
    return render_template('learn.html', categories=categories)

@app.route('/learn/<category>')
def learn_category(category):
    # This would typically fetch flower information from the database
    # For now, we'll use placeholder data
    flowers = FlowerInfo.query.filter(
        FlowerInfo.name.like(f'%{category.lower().rstrip("s")}%')
    ).all()
    
    return render_template(
        'learn_category.html',
        category=category,
        flowers=flowers
    )
if __name__ == '__main__':
    with app.app_context():
        db.drop_all()  # Drop all tables
        db.create_all()  # Create all tables
    app.run(debug=True)



















# from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import os
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Database configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# db = SQLAlchemy(app)

# # User model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(150), nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)

# # Load the trained model
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
#     img = image.resize((128, 128))
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
#     if 'user_id' not in session:
#         flash('Please log in to access this page.')
#         return redirect(url_for('login'))

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

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         username = request.form['username']
#         password = request.form['password']
#         hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
        
#         new_user = User(name=name, email=email, username=username, password=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
        
#         flash('Account created successfully! Please log in.')
#         return redirect(url_for('login'))
#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
        
#         user = User.query.filter_by(username=username).first()
        
#         if user and check_password_hash(user.password, password):
#             session['user_id'] = user.id
#             flash('Login successful!')
#             return redirect(url_for('home'))
#         else:
#             flash('Invalid credentials. Please try again.')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.pop('user_id', None)
#     flash('You have been logged out.')
#     return redirect(url_for('home'))

# if __name__ == '__main__':
#     with app.app_context():
#         db.drop_all()  # Drop all tables
#         db.create_all()  # Create all tables
#     app.run(debug=True)