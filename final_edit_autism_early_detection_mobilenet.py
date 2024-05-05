#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import mediapipe as mp
import time
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model(r"C:\Users\Abdalrahman\Downloads\model_autism.h5")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

def preprocess_image(img):
    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    reshaped_img = normalized_img.reshape((1, 224, 224, 3))
    return reshaped_img

def count_fingers(landmarks):
    thumb_tip = 4
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20
    fingers_up = 0
    
    if landmarks[thumb_tip].y < landmarks[thumb_tip - 1].y:
        fingers_up += 1

    for finger_tip, prev_joint in zip([index_tip, middle_tip, ring_tip, pinky_tip], [7, 11, 15, 19]):
        if landmarks[finger_tip].y < landmarks[prev_joint].y:
            fingers_up += 1

    return fingers_up

def process_frame():
    cap = cv2.VideoCapture(0)
    finger_counts = []
    threshold = 5  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                finger_count = count_fingers(hand_landmarks.landmark)
                finger_counts.append(finger_count)
                if len(finger_counts) > threshold:
                    finger_counts.pop(0) 
                    if all(count in [1, 0, 5] for count in finger_counts):
                        cv2.putText(frame, "Normal Kid", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Autism Kid", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_realtime():
    cap = cv2.VideoCapture(0)
    prediction_threshold = 0.5  # Threshold for prediction values
    predictions = []  # List to store predictions

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        normal='normal'
        cv2.putText(frame, normal, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        autism='autism'
        cv2.putText(frame, autism, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                input_data = preprocess_image(face_img)
                prediction = model.predict(input_data)[0][0]  # Extract single value
                predictions.append(prediction)
                print (predictions)
                if len(predictions) == 20:
                    autism_count = sum(pred > prediction_threshold for pred in predictions)
                    normal_count = len(predictions) - autism_count
                    cv2.putText(frame, str(autism_count), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, str(normal_count), (300, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, str(autism_count), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, str(normal_count), (300, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, str(autism_count), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, str(normal_count), (300, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if normal_count > autism_count:
                        predicted_class = "this is normal child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        predicted_class = "this is normal child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        predicted_class = "this is normal child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        predicted_class = "this is autism child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        predicted_class = "this is autism child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        predicted_class = "this is autism child"
                        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    time.sleep(5)
                    predictions = []
                    
        cv2.imshow('Real-Time Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_recognition():
    messagebox.showinfo("Instructions", "Please show your hand.")
    process_frame()
    messagebox.showinfo("Result", "Hand gesture recognition completed.")

def start_realtime_prediction():
    messagebox.showinfo("Instructions", "Please show your face for real-time prediction.")
    process_realtime()
    messagebox.showinfo("Result", "Real-time prediction completed.")

def open_autism_system():
    new_window = tk.Toplevel(root)
    new_window.title("Autism Prediction System")

    screen_width = new_window.winfo_screenwidth()
    screen_height = new_window.winfo_screenheight()

    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    new_window.geometry(f"{window_width}x{window_height}")

    background_image = Image.open(r"E:\autism_project\gui2.png")
    background_image = background_image.resize((window_width, window_height), Image.ANTIALIAS)
    background_photo = ImageTk.PhotoImage(background_image)
    
    background_label = tk.Label(new_window, image=background_photo)
    background_label.image = background_photo  
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    btn_recognition = tk.Button(new_window, text="Start Hand Gesture Recognition", command=start_recognition, bg='white', bd=4, relief='raised', font=('Helvetica', 14), width=30)
    btn_recognition.place(relx=0.25, rely=0.4, anchor='center')

    btn_realtime_prediction = tk.Button(new_window, text="Start Real-time Prediction", command=start_realtime_prediction, bg='white', bd=4, relief='raised', font=('Helvetica', 14), width=30)
    btn_realtime_prediction.place(relx=0.75, rely=0.4, anchor='center')

root = tk.Tk()
root.title("Main Page")
click_image = Image.open(r"E:\autism_project\click.png")
click_image = click_image.resize((160, 50), Image.ANTIALIAS)
click_photo = ImageTk.PhotoImage(click_image)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = int(screen_width * 0.5)
window_height = int(screen_height * 0.5)
root.geometry(f"{window_width}x{window_height}")


background_image_main = Image.open(r"E:\autism_project\gui1.png")
background_image_main = background_image_main.resize((window_width, window_height), Image.ANTIALIAS)
background_photo_main = ImageTk.PhotoImage(background_image_main)
background_label_main = tk.Label(root, image=background_photo_main)
background_label_main.place(x=0, y=0, relwidth=1, relheight=1)

def open_autism_system():
    new_window = tk.Toplevel(root)
    new_window.title("Autism Prediction System")
    
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    new_window.geometry(f"{window_width}x{window_height}")

    background_image = Image.open(r"E:\autism_project\gui2.png")
    background_image = background_image.resize((window_width, window_height), Image.ANTIALIAS)
    background_photo = ImageTk.PhotoImage(background_image)

    background_label = tk.Label(new_window, image=background_photo)
    background_label.image = background_photo  
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    click_image1 = Image.open(r"E:\autism_project\click2.png")
    click_image1 = click_image1.resize((350, 100), Image.ANTIALIAS)
    click_photo1 = ImageTk.PhotoImage(click_image1)

    click_image2 = Image.open(r"E:\autism_project\click1.png")
    click_image2 = click_image2.resize((350, 120), Image.ANTIALIAS)
    click_photo2 = ImageTk.PhotoImage(click_image2)

    btn_realtime_recognition = tk.Button(new_window, image=click_photo1,bd=0, bg='#1B2838',  command=start_recognition)
    btn_realtime_recognition.image = click_photo1
    btn_realtime_recognition.place(relx=0.2, rely=0.8, anchor='center')

    btn_realtime_prediction = tk.Button(new_window, image=click_photo2, command=start_realtime_prediction,bd=0, bg='#1B2838')
    btn_realtime_prediction.image = click_photo2
    btn_realtime_prediction.place(relx=0.8, rely=0.8, anchor='center')

btn_open_autism_system = tk.Button(root, image=click_photo, bd=0, bg='#1B2838', relief='flat', command=open_autism_system)
btn_open_autism_system.image = click_photo 
btn_open_autism_system.place(relx=0.05, rely=0.95, anchor='sw')
btn_open_autism_system.bind("<Button-1>", open_autism_system)

root.mainloop()


# In[ ]:




