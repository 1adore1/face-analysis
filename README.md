# Real-Time Face Analysis Application
![](https://github.com/1adore1/face-analysis/blob/main/gif.gif)
### Overview
This application detects faces in real-time using a webcam and predicts gender, age range, and emotional state based on the face captured. It uses pre-trained models for age, gender, and emotion recognition and displays the results on the video stream.

### Features
* **Gender Recognition**: Classifies the detected face as Male or Female.
* **Age Estimation**: Predicts the age of the person and maps it to an age range.
* **Emotion Detection**: Recognizes one of the following emotions: Angry, Happy, Sad, Surprise, Neutral.
  
### Installation
1. Clone the repository:
```
git clone https://github.com/1adore1/face-analysis.git
cd face-analysis
```
2. Install required libraries:
```
pip install opencv-python torch numpy
```
### Usage
1. Run the application:
```
python face_analysis.py
```
2. The application will open a video stream from your webcam. Detected faces will be annotated with:
* **Predicted gender** (Male/Female)
* **Estimated age range** (e.g., 21-25, 36-40)
* **Recognized emotion** (Angry, Happy, Sad, Surprise, Neutral)
3. Quit the application by pressing the ```q``` key.
