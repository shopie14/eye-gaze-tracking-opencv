import cv2
import numpy as np

# Load Haarcascade Classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect face and eyes
def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            if ey < h/2:  # Conditional filter to prevent false detection
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                pupil_center = detect_pupil(roi_gray[ey:ey+eh, ex:ex+ew], roi_color[ey:ey+eh, ex:ex+ew])
                if pupil_center:
                    gaze_direction = estimate_gaze(pupil_center, roi_color[ey:ey+eh, ex:ex+ew])
                    cv2.putText(frame, gaze_direction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# Function to detect the pupil using centroid method
def detect_pupil(eye_gray, eye_color):
    # Apply threshold
    _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.circle(eye_color, (x + int(w/2), y + int(h/2)), 5, (0, 0, 255), -1)
        return (x + int(w/2), y + int(h/2))
    return None

# Placeholder for Hough Circle Transform and Blob Detection methods
# def detect_pupil_hough(eye_gray, eye_color):
#     # Add Hough Circle Transform code here
#     pass
# def detect_pupil_blob(eye_gray, eye_color):
#     # Add Blob Detection code here
#     pass

# Function to estimate gaze direction
def estimate_gaze(pupil_center, eye_frame):
    eye_center = (eye_frame.shape[1] // 2, eye_frame.shape[0] // 2)
    if pupil_center:
        dx = pupil_center[0] - eye_center[0]
        dy = pupil_center[1] - eye_center[1]
        horizontal_ratio = pupil_center[0] / eye_frame.shape[1]
        vertical_ratio = pupil_center[1] / eye_frame.shape[0]

        if horizontal_ratio < 0.4:
            horizontal_direction = 'Left'
        elif horizontal_ratio > 0.6:
            horizontal_direction = 'Right'
        else:
            horizontal_direction = 'Center'

        if vertical_ratio < 0.4:
            vertical_direction = 'Top'
        elif vertical_ratio > 0.6:
            vertical_direction = 'Bottom'
        else:
            vertical_direction = 'Center'

        return f'{vertical_direction}-{horizontal_direction}'
    return 'Unknown'

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_face_and_eyes(frame)
        cv2.imshow('Gaze Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
