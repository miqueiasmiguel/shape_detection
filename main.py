import cv2
import numpy as np
from red_detection import red_detection
from glass_detection import glass_detection


N = 10
i = 0
zx = 0
zy = 0
zix = 0
ziy = 0
p = [[0, 0], [0, 0], [0, 0], [0, 0]]
pi = [[0, 0], [0, 0], [0, 0], [0, 0]]
array_old = [[0, 0], [0, 0], [0, 0], [0, 0]]


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 160)
cv2.createTrackbar("Threshold1", "Parameters", 33, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 31, 255, empty)
cv2.createTrackbar("Area Min", "Parameters", 40994, 200000, empty)
cv2.createTrackbar("Area Max", "Parameters", 60455, 200000, empty)

cap = cv2.VideoCapture(0 , apiPreference=cv2.CAP_DSHOW)

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
    area_min = cv2.getTrackbarPos("Area Min", "Parameters")
    area_max = cv2.getTrackbarPos("Area Max", "Parameters")

    cx, cy = red_detection(frame)
    array, g_success = glass_detection(frame, threshold1, threshold2, area_min, area_max)

    if g_success == False:
        array = array_old
    else:
        array_old = array

    zix = zix + cx
    ziy = ziy + cy

    pi[0][0] = pi[0][0] + array[0][0]
    pi[0][1] = pi[0][1] + array[0][1]
    pi[1][0] = pi[1][0] + array[1][0]
    pi[1][1] = pi[1][1] + array[1][1]
    pi[2][0] = pi[2][0] + array[2][0]
    pi[2][1] = pi[2][1] + array[2][1]
    pi[3][0] = pi[3][0] + array[3][0]
    pi[3][1] = pi[3][1] + array[3][1]

    if i >= N:
        zx = int(zix/N)
        zy = int(ziy/N)

        p[0][0] = int(pi[0][0]/N)
        p[0][1] = int(pi[0][1]/N)
        p[1][0] = int(pi[1][0]/N)
        p[1][1] = int(pi[1][1]/N)
        p[2][0] = int(pi[2][0]/N)
        p[2][1] = int(pi[2][1]/N)
        p[3][0] = int(pi[3][0]/N)
        p[3][1] = int(pi[3][1]/N)

        zix = 0
        ziy = 0
        pi = [[0, 0], [0, 0], [0, 0], [0, 0]]

        i = 0

        print(
            f"P1({p[0][0] - zx},{p[0][1] - zy}), P2({p[1][0] - zx},{p[1][1] - zy}), P3({p[2][0] - zx},{p[2][1] - zy}), P4({p[3][0] - zx},{p[3][1] - zy})")

    cv2.circle(frame, (zx, zy), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"zero({zx},{zy})", (zx - 20, zy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.circle(frame, (p[0][0], p[0][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P1({p[0][0] - zx},{p[0][1] - zy})", (p[0][0] - 20, p[0][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.circle(frame, (p[1][0], p[1][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P2({p[1][0] - zx},{p[1][1] - zy})", (p[1][0] - 20, p[1][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.circle(frame, (p[2][0], p[2][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P3({p[2][0] - zx},{p[2][1] - zy})", (p[2][0] - 20, p[2][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.circle(frame, (p[3][0], p[3][1]), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"P4({p[3][0] - zx},{p[3][1] - zy})", (p[3][0] - 20, p[3][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    frame_final = cv2.polylines(img=frame, pts=np.int32([p]), isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow('frame_final', frame_final)

    i = i + 1

    if cv2.waitKey(1) == ord('q'):
        cap.release()
