import numpy as np
import cv2


def red_detection(frame):

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(frame_hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(frame_hsv, (175, 50, 20), (180, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx = 0
    cy = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03*perimeter, True)
        if len(approx) == 4 and area >= 200:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

    cv2.imshow('mask', mask)

    return cx, cy