import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Define paths
path = 'C:\\Users\\amrit\\Downloads\\archive (1)\\dataset\\'
no_tumor_path = os.path.join(path, 'no\\')
yes_tumor_path = os.path.join(path, 'yes\\')
input_size = 64

# Initialize dataset and labels
dataset = []
label = []

# Load no tumor images
for image_name in os.listdir(no_tumor_path):
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(no_tumor_path, image_name))
        image = Image.fromarray(image, 'RGB').resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(0)

# Load yes tumor images
for image_name in os.listdir(yes_tumor_path):
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(yes_tumor_path, image_name))
        image = Image.fromarray(image, 'RGB').resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(1)

# Convert dataset and labels to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(input_size, input_size, 3)),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('braintumor_best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=16),
          epochs=15, 
          validation_data=(x_test, y_test), 
          callbacks=[early_stopping, model_checkpoint],
          shuffle=True)


loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

# Print the accuracy
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')