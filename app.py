from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configurations
UPLOAD_FOLDER = 'uploads/'  # Directory to store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/model.h5'  # Path to the trained model

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    """Preprocess and predict the disease of the uploaded image."""
    img = image.load_img(filepath, target_size=(224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Get class with highest probability
    return class_idx

@app.route('/')
def index():
    """Render the homepage with mango disease information."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and predict disease."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict disease from uploaded image
        class_idx = predict_image(filepath)

        # Map class indices to disease information
        class_labels = {
            0: ('Anthracnose', 
                'A fungal disease causing black lesions on mango leaves and fruits.', 
                'Use fungicide containing copper oxychloride or carbendazim. Prune affected parts.'),
            1: ('Bacterial Canker', 
                'A bacterial infection causing lesions and dieback.', 
                'Spray copper-based bactericides. Avoid injuries to the plant.'),
            2: ('Cutting Weevil', 
                'A pest that damages mango stems and fruits.', 
                'Use insecticides such as chlorpyrifos. Destroy affected fruits.'),
            3: ('Die Back', 
                'A fungal disease causing branches to wither and die.', 
                'Apply fungicides like thiophanate-methyl. Improve air circulation.'),
            4: ('Gall Midge', 
                'Insects that form galls on leaves, reducing yield.', 
                'Use systemic insecticides like imidacloprid. Destroy affected parts.'),
            5: ('Healthy', 
                'No disease detected.', 
                'Your plant appears healthy! Maintain regular care.'),
            6: ('Powdery Mildew', 
                'A fungal disease causing white, powdery growth on leaves and fruits.', 
                'Spray wettable sulfur or potassium bicarbonate.'),
            7: ('Sooty Mould', 
                'A fungal disease causing black, sooty patches on leaves.', 
                'Control sap-sucking pests like aphids. Wash leaves with soap solution.')
        }

        disease_name, disease_description, disease_cure = class_labels[class_idx]

        return render_template(
    'result.html',
    disease_name=disease_name,
    disease_description=disease_description,
    disease_cure=disease_cure,
    uploaded_image_url=url_for('static', filename='uploads/' + filename)
)

    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Ensure the uploads folder exists
    app.run(debug=True)
