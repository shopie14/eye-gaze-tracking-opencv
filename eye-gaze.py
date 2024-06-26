import cv2
import numpy as np
import math
import time

font = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_SOURCE = 0
BLINK_LIMIT = 5.2  # should be changed
HEIGHT = 720
WIDTH = 1280
blinked = False
blinkedMinValue = 2

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

def detect_pupil(eye_gray):
    # Hough Circle Transform
    circles = cv2.HoughCircles(eye_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            return (x, y)
    # Blob Detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1500
    params.filterByConvexity = True
    params.minConvexity = 0.8
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(eye_gray)
    if keypoints:
        return (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))
    # Centroid Detection
    _, thresh = cv2.threshold(eye_gray, 55, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None

def blink_detection(eye_gray):
    _, threshold_eye = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    return left_side_white + right_side_white

def get_gaze_ratio(eye_gray):
    _, threshold_eye = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

def get_gaze_ratio_updown(eye_gray):
    _, threshold_eye = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    up_side_threshold = threshold_eye[0: int(height / 2), 0: width]
    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)
    if up_side_white == 0:
        gaze_ratio = 1
    elif down_side_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = up_side_white / down_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (216, 255, 158), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]

            pupil = detect_pupil(eye_gray)
            if pupil is not None:
                cv2.circle(eye_color, pupil, 2, (0, 255, 0), -1)

            blink_ratio = blink_detection(eye_gray)
            if blink_ratio > BLINK_LIMIT:
                cv2.putText(frame, "Blinked ", (50, 300), font, 2, (255, 255, 255), 2)
                if not blinked:
                    blinked = True
                    blinkedSec = time.time()
                if abs(time.time() - blinkedSec) > blinkedMinValue:
                    blinkedSec = time.time()
                    print("komut6")
            else:
                blinkedSec = time.time()
                blinked = False

            gaze_ratio_left = get_gaze_ratio(eye_gray)
            gaze_ratio_updown = get_gaze_ratio_updown(eye_gray)

            if gaze_ratio_left <= 0.65:
                cv2.putText(frame, "RIGHT "+str("%.2f" % gaze_ratio_left), (50, 100), font, 2, (0, 0, 255), 3)
                new_frame[:] = (0, 0, 255)
            elif 0.65 < gaze_ratio_left <= 1.8:
                cv2.putText(frame, "CENTER "+str("%.2f" % gaze_ratio_left), (50, 100), font, 2, (0, 0, 255), 3)
            elif 1.8 < gaze_ratio_left <= 5:
                new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "LEFT "+str("%.2f" % gaze_ratio_left), (50, 100), font, 2, (0, 0, 255), 3)

            if gaze_ratio_updown > 1:
                cv2.putText(frame, "DOWN "+str("%.2f" % gaze_ratio_updown), (50, 500), font, 2, (0, 255, 0), 3)
            elif gaze_ratio_updown < 1:
                cv2.putText(frame, "UP "+str("%.2f" % gaze_ratio_updown), (50, 600), font, 2, (255, 0, 0), 3)

    cv2.imshow("Footage", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
