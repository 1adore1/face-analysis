import cv2
from PIL import Image
import torch
import torchvision
import numpy as np

emotion_model = torch.load('emotion_model_old.pth', map_location=torch.device('cpu'))
emotion_model.eval()

age_model = torch.load('age_model_old.pth', map_location=torch.device('cpu'))
age_model.eval()

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)
faces_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = torch.from_numpy(face).unsqueeze(0) / 255
        face = face.unsqueeze(0).float()

        emotion = classes[emotion_model(face).argmax()]
        age = str(int(age_model(face)))

        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, age, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, age, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()