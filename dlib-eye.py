import cv2
import dlib
import numpy as np

# Load Dlib's pre-trained shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect face and eyes
def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        landmarks = predictor(gray, face)
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]

        # Convert landmarks to numpy arrays
        left_eye = np.array([[p.x, p.y] for p in left_eye])
        right_eye = np.array([[p.x, p.y] for p in right_eye])

        # Draw rectangles around the eyes
        for eye in [left_eye, right_eye]:
            x, y, w, h = cv2.boundingRect(eye)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            eye_gray = gray[y:y+h, x:x+w]
            eye_color = frame[y:y+h, x:x+w]
            pupil_center = detect_pupil(eye_gray, eye_color)
            if pupil_center:
                gaze_direction = estimate_gaze(pupil_center, eye_color)
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

# Function to detect the pupil using Hough Circle Transform
def detect_pupil_hough(eye_gray, eye_color):
    blurred_eye = cv2.medianBlur(eye_gray, 5)
    circles = cv2.HoughCircles(blurred_eye, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(eye_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(eye_color, (i[0], i[1]), 2, (0, 0, 255), 3)
            return (i[0], i[1])
    return None

# Function to detect the pupil using Blob Detection
def detect_pupil_blob(eye_gray, eye_color):
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1500
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(eye_gray)
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        cv2.circle(eye_color, (x, y), int(keypoint.size / 2), (0, 255, 0), 2)
        cv2.circle(eye_color, (x, y), 2, (0, 0, 255), 3)
        return (x, y)
    return None

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
