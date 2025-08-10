import cv2
import numpy as np

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to guess emotion (simple logic based on mouth openness)
def guess_emotion(gray_face):
    h, w = gray_face.shape
    mouth_region = gray_face[int(h/2):, :]
    blur = cv2.GaussianBlur(mouth_region, (15, 15), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(thresh == 255)
    
    if white_pixels > 2500:
        return "Happy"
    elif white_pixels < 1000:
        return "Sad"
    else:
        return "Neutral"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        emotion = guess_emotion(face_gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
