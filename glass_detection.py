from asyncio.windows_events import NULL
import numpy as np
import cv2


def glass_detection(frame, threshold1: int, threshold2: int, area_min: int, area_max: int):

    frame_blur = cv2.GaussianBlur(frame, (7,7), 1)
    frame_grey = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_canny = cv2.Canny(frame_grey, threshold1, threshold2)

    kernel = np.ones((5,5))
    frame_dil = cv2.dilate(frame_canny, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(frame_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours == NULL:
        success = False
    else:
        success = True

    points = [0,0,0,0,0,0,0,0]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min and area < area_max:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            if len(approx) == 4:
                points = approx.ravel()
                

    array = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    
    cv2.imshow('frame_canny', frame_canny)
    cv2.imshow('frame_dil', frame_dil)

    return array, success