import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained CNN model
model = load_model('braintumor_best_model.keras')
print('Model loaded. Check http://127.0.0.1:5000/')

# Recommendations for tumor detection
recommendations = {
    'Tumor': {
        'Best_Treatment': 'Chemo and Radiation',
        'Hospital': 'Fortis Hospital, Gurugram, Haryana',
        'Doctor': 'Dr. Sandeep Vaishy',
        'Cost': 'INR 2 to 6 lakhs'
    }
}

# Load the dataset containing hospital, treatment, and doctor information
df = pd.read_csv('hospital.csv', encoding='latin1')

# Function to get the class name (No Tumor / Tumor)
def get_className(classNo):
    if classNo == 0:
        return "No, you do not have a brain tumor."
    elif classNo == 1:
        return "Yes, you have a brain tumor."

# Function to process the image and get prediction
def getResult(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
    result = model.predict(image)
    return np.argmax(result, axis=1)[0]

# Function to get treatment, hospital, and doctor recommendations based on disease name
def get_recommendations(disease_name):
    df.columns = df.columns.str.strip()  # Strip spaces from column names
    disease_name = disease_name.strip()

    # Fetch recommendations from the dataset
    disease_data = df[df['Disease'].str.contains(disease_name, case=False, na=False)]
    
    if disease_data.empty:
        return None

    treatments = disease_data['Best Treatement'].values[0].split(';')
    hospitals = disease_data['Hospital'].values[0].split(';')
    doctors = disease_data['Specialist Doctor'].values[0].split(';')

    return {
        'Disease': disease_name,
        'Treatments': treatments,
        'Hospitals': hospitals,
        'Doctors': doctors
    }

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home page (Upload MRI or Enter Disease Name)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Handle predictions and recommendations
@app.route('/predict', methods=['POST'])
def upload_or_recommend():
    disease_name = request.form.get('disease_name', '').strip()

    # Check if an image is uploaded
    if 'file' in request.files and request.files['file'].filename != '':
        # Process uploaded MRI image
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        # Get tumor prediction result
        value = getResult(file_path)
        prediction = get_className(value)

        # If tumor is detected, show the recommendations
        if value == 1:
            tumor_recommendations = recommendations['Tumor']
        else:
            tumor_recommendations = None

        # Pass filename to display uploaded image
        return render_template('result.html', 
                               prediction=prediction, 
                               recommendations=tumor_recommendations, 
                               filename=filename)

    # Check if a disease name is entered
    elif disease_name != '':
        disease_recommendations = get_recommendations(disease_name)

        if disease_recommendations:
            return render_template('result.html', 
                                   prediction=f"Recommendations for {disease_name}:", 
                                   recommendations=disease_recommendations)
        else:
            return render_template('result.html', 
                                   prediction=f"No recommendations found for '{disease_name}'.")

    # If neither an image nor a disease name is provided, show an error
    return render_template('index.html', 
                           error="Please upload an image or enter a disease name.", 
                           disease_name=disease_name)  # Retain entered disease name in the form

if __name__ == '__main__':
    app.run(debug=True)
