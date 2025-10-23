# train_model.py
import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Define directories
DATA_DIR = "data"
IMAGES_DIR = "images"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

image_data = []
labels = []

# Loop through all subdirectories in the images directory
print("Loading images from subdirectories...")
for person_name in os.listdir(IMAGES_DIR):
    person_dir = os.path.join(IMAGES_DIR, person_name)
    if os.path.isdir(person_dir):
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (100, 100))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data.append(image)
                labels.append(person_name)

if not image_data:
    print("No images found. Run collect_data.py first.")
else:
    # Preprocess the data
    images = np.array(image_data, dtype='float32') / 255.0
    images = images.reshape(images.shape[0], 100, 100, 1)

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(integer_labels)

    with open(os.path.join(DATA_DIR, 'label_encoder.p'), 'wb') as f:
        pickle.dump(label_encoder, f)

    X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)
    num_classes = len(label_encoder.classes_)

    print(f"Found {len(image_data)} images for {num_classes} people.")

    # Build the CNN model
    print("Building model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    model.save("final_model.h5")
    print("Model trained and saved as final_model.h5")