import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load Dlib's pre-trained shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect face, eyes, and yawns
def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    print("Number of faces detected: {}".format(len(faces)))
    
    for idx, face in enumerate(faces):
        print(f"Detection {idx}")
        print(f'Left: {face.left()} Top: {face.top()} Right: {face.right()} Bottom: {face.bottom()}')
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        landmarks = predictor(gray, face)
        
        # Detect eyes
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        left_eye = np.array([[p.x, p.y] for p in left_eye])
        right_eye = np.array([[p.x, p.y] for p in right_eye])

        print("Number of pair of eyes detected: {}".format(len([left_eye, right_eye])))
        
        # Draw rectangles around the eyes
        for eye in [left_eye, right_eye]:
            x, y, w, h = cv2.boundingRect(eye)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eye_gray = gray[y:y + h, x:x + w]
            eye_color = frame[y:y + h, x:x + w]
            pupil_center = detect_pupil(eye_gray, eye_color)
            if pupil_center:
                gaze_direction, displacement = estimate_gaze(pupil_center, eye_color)
                cv2.putText(frame, gaze_direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detect yawn
        upper_lip = landmarks.part(62)
        lower_lip = landmarks.part(66)
        lip_distance = distance.euclidean((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))
        yawn_threshold = 20  # This value may need to be adjusted based on experimentation
        if lip_distance > yawn_threshold:
            cv2.putText(frame, "Yawning", (face.left(), face.bottom() + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Yawn detected")

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

        gaze_direction = f'{vertical_direction}-{horizontal_direction}'
        displacement = (dx, dy)
        return gaze_direction, displacement
    return 'Unknown', (0, 0)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_face_and_eyes(frame)
        cv2.imshow('Gaze and Yawn Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
