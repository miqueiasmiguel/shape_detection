import cv2
import numpy as np
from red_detection import red_detection
from glass_detection import glass_detection

N = 10
i = 0
p = [[0, 0], [0, 0], [0, 0], [0, 0]]
pi = [[0, 0], [0, 0], [0, 0], [0, 0]]

cap = cv2.VideoCapture(2, apiPreference=cv2.CAP_DSHOW)


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 160)
cv2.createTrackbar("Threshold1", "Parameters", 72, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 15, 255, empty)
cv2.createTrackbar("Area Min", "Parameters", 250000, 250000, empty)
cv2.createTrackbar("Area Max", "Parameters", 260000, 300000, empty)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    threshold3 = cv2.getTrackbarPos("Area Min", "Parameters")
    threshold4 = cv2.getTrackbarPos("Area Max", "Parameters")

    frame_blur = cv2.GaussianBlur(frame, (7, 7), 1)
    frame_grey = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_canny = cv2.Canny(frame_grey, threshold1, threshold2)

    kernel = np.ones((5, 5))
    frame_dil = cv2.dilate(frame_canny, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        frame_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    points = [0, 0, 0, 0, 0, 0, 0, 0]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold3 and area < threshold4:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            if len(approx) == 4:
                points = approx.ravel()

    array = np.array([[points[0], points[1]], [points[2], points[3]], [
                     points[4], points[5]], [points[6], points[7]]])

    pi[0][0] = pi[0][0] + array[0][0]
    pi[0][1] = pi[0][1] + array[0][1]
    pi[1][0] = pi[1][0] + array[1][0]
    pi[1][1] = pi[1][1] + array[1][1]
    pi[2][0] = pi[2][0] + array[2][0]
    pi[2][1] = pi[2][1] + array[2][1]
    pi[3][0] = pi[3][0] + array[3][0]
    pi[3][1] = pi[3][1] + array[3][1]

    if i >= N:
        p[0][0] = int(pi[0][0]/N)
        p[0][1] = int(pi[0][1]/N)
        p[1][0] = int(pi[1][0]/N)
        p[1][1] = int(pi[1][1]/N)
        p[2][0] = int(pi[2][0]/N)
        p[2][1] = int(pi[2][1]/N)
        p[3][0] = int(pi[3][0]/N)
        p[3][1] = int(pi[3][1]/N)

        pi = [[0, 0], [0, 0], [0, 0], [0, 0]]
        i = 0

        print(p)


    cv2.circle(frame, (p[0][0], p[0][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P1({p[0][0]},{p[0][1]})", (p[0][0] + 20, p[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (p[1][0], p[1][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P2({p[1][0]},{p[1][1]})", (p[1][0] - 20, p[1][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (p[2][0], p[2][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P3({p[2][0]},{p[2][1]})", (p[2][0] - 20, p[2][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (p[3][0], p[3][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P4({p[3][0]},{p[3][1]})", (p[3][0] - 20, p[3][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_final = cv2.polylines(img=frame, pts=np.int32(
        [p]), isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow('frame_final', frame_final)

    cv2.imshow('frame_grey', frame_grey)
    cv2.imshow('frame_canny', frame_canny)
    cv2.imshow('frame_dil', frame_dil)

    i = i + 1

    if cv2.waitKey(1) == ord('q'):
        cap.release()
