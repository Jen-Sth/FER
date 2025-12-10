# pip install streamlit opencv-python torch torchvision numpy
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model exactly as trained
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 emotions

# Load weights
model.load_state_dict(torch.load("emotion_resnet18.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Real-Time Facial Emotion Detection")
run = st.checkbox('Start Webcam')

# Transform for face images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            pil_img = Image.fromarray(face_img)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                emotion = class_names[pred.item()]

            # Draw rectangle and label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Show frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    if cap is not None:
        cap.release()

# to run: streamlit run app.py
