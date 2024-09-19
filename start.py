import cv2
import numpy as np
import dlib
from pathlib import Path
from imutils import face_utils
import pygame  
import threading

pygame.mixer.init()

alarm_sound = pygame.mixer.Sound("alarm.mp3")

cap = cv2.VideoCapture(0)

# Initializing face detector
detector = dlib.get_frontal_face_detector()

# 68 face landmarks (example for detection)
predictor = dlib.shape_predictor("shape_predictor.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to check if the eye is open or closed
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    
    # Checking the blinking ratio
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

# Function to play alarm sound in a separate thread
def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play(-1) 

# Function to stop the alarm sound
def stop_alarm():
    if pygame.mixer.get_busy():
        alarm_sound.stop()

alarm_on = False  # To track if the alarm is already playing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)

    # Check if faces are detected
    if len(faces) > 0:
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            
            face_frame = frame.copy()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "YOU ARE SLEEPING!!, ALARM IS ACTIVATED"
                    color = (0,0,255)
                    if not alarm_on:  # Play alarm if it's not already playing
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            
            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                drowsy += 1
                active = 0
                if drowsy > 6:
                    status = "ATTENTION PLEASE: I THINK YOU ARE TIRED"
                    color = (255,165,0)
                    if alarm_on:
                        stop_alarm()  # Stop alarm when no longer sleeping
                        alarm_on = False
            
            else:
                sleep = 0
                drowsy = 0
                active += 1
                if active > 6:
                    status = "All Good :)"
                    color = (135,206,235)
                    if alarm_on:
                        stop_alarm()  # Stop alarm when no longer sleeping
                        alarm_on = False

            cv2.putText(frame, status, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Drawing landmarks on the face
            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 0, 0), -1)
        
        cv2.imshow("Result of detector", face_frame)  # Show the face frame only if a face is detected
    
    cv2.imshow("Frame", frame)  # Show the main frame (even if no faces are detected)

    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
