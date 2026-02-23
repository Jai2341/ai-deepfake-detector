import cv2
import numpy as np
from model.predict import predict_image

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames_checked = 0
    fake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frames_checked % 30 == 0:   # check every 30th frame
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            label, conf = predict_image(temp_path)

            if label == "FAKE":
                fake_count += 1

        frames_checked += 1

    cap.release()

    if fake_count > 3:
        return "FAKE", 85
    else:
        return "REAL", 90