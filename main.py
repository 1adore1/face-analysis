import cv2
import torch

def age_to_range(age):
    if 0 <= age <= 5:
        return '0-5'
    elif 6 <= age <= 10:
        return '6-10'
    elif 11 <= age <= 15:
        return '11-15'
    elif 16 <= age <= 20:
        return '16-20'
    elif 21 <= age <= 25:
        return '21-25'
    elif 26 <= age <= 30:
        return '26-30'
    elif 31 <= age <= 35:
        return '31-35'
    elif 36 <= age <= 40:
        return '36-40'
    elif 41 <= age <= 45:
        return '41-45'
    elif 46 <= age <= 50:
        return '46-50'
    elif 51 <= age <= 55:
        return '51-55'
    elif 56 <= age <= 60:
        return '56-60'
    elif 61 <= age <= 65:
        return '61-65'
    elif age >= 65:
        return '65+'
    else:
        return 'Unknown'

age_model = torch.load('models/age_model.pth', map_location=torch.device('cpu'))
age_model.eval()

gender_model = torch.load('models/gender_model.pth', map_location=torch.device('cpu'))
gender_model.eval()

emotion_model = torch.load('models/emotion_model.pth', map_location=torch.device('cpu'))
emotion_model.eval()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
genders = ['Male', 'Female']

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

        age = int(age_model(face))
        age_range = age_to_range(age)
        gender = genders[gender_model(face).argmax()]
        emotion = emotions[emotion_model(face).argmax()]

        cv2.putText(frame, f'{gender} {age_range}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, f'{gender} {age_range}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, emotion, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, emotion, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()