import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import math

# Load Model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('best.pt').to(device)

# Title
st.title("Aplikasi Deteksi Jumlah Kehadiran Siswa")

# Sidebar for selecting input source
st.sidebar.title("Pilih Sumber Video")
input_source = st.sidebar.radio("Sumber:", ("Unggah Video", "Webcam"))

if input_source == "Unggah Video":
    uploaded_video = st.file_uploader("Unggah video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()  # placeholder for video frames

        classNames = ["isi", "kosong"]
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run YOLO model on the frame
                results = model.track(frame, persist=True, conf=0.2)

                # Initialize counters
                empty_seats = 0
                present_students = 0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if box.id is not None:
                            # Bounding Box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # ID
                            id_object = int(box.id[0])

                            # Confidence
                            conf = math.ceil((box.conf[0] * 100)) / 100

                            # Class Name
                            cls = int(box.cls[0])
                            currentClass = classNames[cls]

                            # Count classes
                            if currentClass == "kosong":
                                empty_seats += 1
                            elif currentClass == "isi":
                                present_students += 1

                            # Warna bounding box
                            color = (0, 255, 0) if currentClass == "kosong" else (0, 0, 255)

                            # Tambahkan data ke bounding box
                            text_2 = f"id: {id_object}, {currentClass}, Conf: {conf}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # Tambahkan jumlah kursi kosong dan mahasiswa hadir di atas frame
                text_kosong = f"Jumlah kursi kosong : {empty_seats}"
                text_hadir = f"Mahasiswa Hadir : {present_students}"

                cv2.putText(frame, text_kosong, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, text_hadir, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the annotated frame in the Streamlit app
                stframe.image(frame, channels="BGR")

            else:
                break

        cap.release()

elif input_source == "Webcam":
    # Open the webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # placeholder for video frames

    classNames = ["isi", "kosong"]
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLO model on the frame
            results = model.track(frame, persist=True, conf=0.2)

            # Initialize counters
            empty_seats = 0
            present_students = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.id is not None:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # ID
                        id_object = int(box.id[0])

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        # Class Name
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]

                        # Count classes
                        if currentClass == "kosong":
                            empty_seats += 1
                        elif currentClass == "isi":
                            present_students += 1

                        # Warna bounding box
                        color = (0, 255, 0) if currentClass == "kosong" else (0, 0, 255)

                        # Tambahkan data ke bounding box
                        text_2 = f"id: {id_object}, {currentClass}, Conf: {conf}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Tambahkan jumlah kursi kosong dan mahasiswa hadir di atas frame
            text_kosong = f"Jumlah kursi kosong : {empty_seats}"
            text_hadir = f"Mahasiswa Hadir : {present_students}"

            cv2.putText(frame, text_kosong, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text_hadir, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame in the Streamlit app
            stframe.image(frame, channels="BGR")

        else:
            break

    cap.release()
