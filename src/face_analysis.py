import cv2
import torch

def age_to_range(age):
    ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), 
              (26, 30), (31, 35), (36, 40), (41, 45), (46, 50), 
              (51, 55), (56, 60), (61, 65)]
    for start, end in ranges:
        if start <= age <= end:
            return f'{start}-{end}'
    return '65+' if age > 65 else 'Unknown'

age_model = torch.load('models/age_model.pth', map_location=torch.device('cpu'), weights_only=False)
age_model.eval()

gender_model = torch.load('models/gender_model.pth', map_location=torch.device('cpu'), weights_only=False)
gender_model.eval()

emotion_model = torch.load('models/emotion_model.pth', map_location=torch.device('cpu'), weights_only=False)
emotion_model.eval()


emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
genders = ['Male', 'Female']

cap = cv2.VideoCapture(0)
faces_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = torch.from_numpy(face).unsqueeze(0) / 255
        face = face.unsqueeze(0).float()

        with torch.no_grad():
            age = int(age_model(face))
            gender = genders[gender_model(face).argmax()]
            emotion = emotions[emotion_model(face).argmax()]

        age_range = age_to_range(age)
        info_text = f'{gender} {age_range}, {emotion}'

        cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
    cv2.imshow('face analysis', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()