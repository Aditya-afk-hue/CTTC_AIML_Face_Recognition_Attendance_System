# recognize.py
import cv2
import numpy as np
from keras.models import load_model
import pickle
import os

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("final_model.h5")

data_dir = os.path.join(os.getcwd(), 'data')
with open(os.path.join(data_dir, 'label_encoder.p'), 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    # No need for equalizeHist if not used in training
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    faces = classifier.detectMultiScale(frame, 1.3, 5)
      
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        prediction = model.predict(preprocess(face), verbose=0)
        pred_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        if confidence > 75: # Confidence Threshold
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            text = f"{pred_label} ({confidence:.1f}%)"
        else:
            text = "Unknown"
        
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()s