import cv2

cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("Cannot open camera")
else:
    print("Camera successfully opened")

cap.release()
