# recognize.py

import cv2
import numpy as np
from keras.models import load_model
import pickle
import os

# Load the pre-trained classifier for face detection
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your trained face recognition model
model = load_model("final_model.h5")

# --- MODIFICATION START ---
# Load the label encoder
data_dir = os.path.join(os.getcwd(), 'data')
with open(os.path.join(data_dir, 'label_encoder.p'), 'rb') as f:
    label_encoder = pickle.load(f)
# --- MODIFICATION END ---

# Initialize video capture from the default system webcam
cap = cv2.VideoCapture(0)

def preprocess(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 100x100 pixels
    img = cv2.resize(img, (100, 100))
    # Apply histogram equalization
    img = cv2.equalizeHist(img)
    # Reshape for the model: (1, height, width, channels)
    img = img.reshape(1, 100, 100, 1)
    # Normalize pixel values
    img = img / 255.0
    return img

# Main loop to capture and process video frames
while True:
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect faces in the frame
    faces = classifier.detectMultiScale(frame, 1.5, 5)
      
    for x, y, w, h in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get the prediction from the model
        prediction = model.predict(preprocess(face))
        pred_index = np.argmax(prediction)
        
        # --- MODIFICATION START ---
        # Convert the prediction index back to a name label
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        # --- MODIFICATION END ---
        
        # Put the predicted label on the frame
        cv2.putText(frame, pred_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()