import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Define class labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the trained model
class EmotionResNet(nn.Module):
    def __init__(self, base_model, num_classes=7):
        super(EmotionResNet, self).__init__()
        self.base = base_model
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)

# Instantiate model
from torchvision.models import resnet18
model = EmotionResNet(resnet18(pretrained=False))
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Real-Time Facial Emotion Recognition")
st.markdown("Turn on your webcam and detect facial emotions in real time.")

run = st.toggle("Start Webcam")

FRAME_WINDOW = st.image([])

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam loop
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_img)
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255, 0, 255), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()