import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
try:
    model = load_model('braintumor_best_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Path to the image for prediction
image_path = 'C:\\Users\\amrit\\Downloads\\archive (1)\\dataset\\pred\\pred2.jpg'

# Define recommendations
recommendations = {
    'Tumor': {
        'Best_Treatment': 'Chemo and Radiation',
        'Hospital': 'Fortis Hospital,Gurugram, Haryana',
        'Doctor': 'Dr.Sandeep Vaishy',
        'Cost': 'INR 2 and 6 lakhs.'
    }
    
}

# Read and preprocess the image
try:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the input size expected by the model
    img = Image.fromarray(image).resize((64, 64))
    img = np.array(img)

    # Normalize the image
    img = img / 255.0

    # Expand dimensions to match the model's input shape
    input_images = np.expand_dims(img, axis=0)

    # Predict the result
    result = model.predict(input_images)

    # Determine the predicted class
    class_label = np.argmax(result, axis=1)[0]

    # Print the predicted result
    if class_label == 0:
        print('No, you do not have a tumor.')
        rec = recommendations['No Tumor']
    else:
        print('Yes, you have a tumor.')
        rec = recommendations['Tumor']

    # Print the recommendations
    print("\nBest Treatment:")
    print(f"- {rec['Best_Treatment']}")
    print("\nHospital:")
    print(f"- {rec['Hospital']}")
    print("\nDoctor:")
    print(f"- {rec['Doctor']}")
    print("\nCost of Treatment:")
    print(f"- {rec['Cost']}")

except Exception as e:
    print(f"Error processing image: {e}")
