# collect_data.py
import cv2
import os

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not os.path.exists("images"):
    os.makedirs("images")

data = []

while len(data) < 100:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    face_points = classifier.detectMultiScale(frame, 1.3, 5)
    
    if len(face_points) > 0:
        for x, y, w, h in face_points:
            face_frame = frame[y:y+h+1, x:x+w+1]
            cv2.imshow("Only face", face_frame)
            
            if len(data) <= 100:
                print(len(data) + 1, "/100")
                data.append(face_frame)
            break

    cv2.putText(frame, str(len(data)) + "/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(30) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
        
if len(data) == 100:
    name = input("Enter Face holder name: ")
    person_dir = os.path.join("images", name)
    os.makedirs(person_dir, exist_ok=True)
    for i in range(100):
        cv2.imwrite(os.path.join(person_dir, name + "_" + str(i) + ".jpg"), data[i])
    print("Data collection done!")
else:
    print("Could not collect 100 samples. Please try again.")