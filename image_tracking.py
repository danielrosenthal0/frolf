import cv2
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image as im

np.set_printoptions(threshold=np.inf)


# Initialize the video capture object
# cap = cv2.VideoCapture('/Users/jackconnelly/Documents/School/Fall_2022/EGR_314/OpenCV/Kendall_5_trimmed_2.MP4')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/Users/jackconnelly/Documents/School/Fall_2022/EGR_314/OpenCV/side_test.mp4')
# cap = cv2.VideoCapture('/Users/jackconnelly/Documents/School/Fall_2022/EGR_314/OpenCV/Kendall_5_trimmed_2.MP4')


# # hsv orange alright?
lower_color = np.array([10, 20, 100])
upper_color = np.array([80, 255, 255])

maxLength = 0
loc = ""
centerX = []
centerY = []
area = []
allVert = []
allNose = []
adjust = 0
prevPos = 0
pos = 0
corner = np.array((0,0))
output = cv2.VideoWriter('sideTest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1340,400))

while True:
# Capture the current frameq
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[300:700, 560:1900]

    # Convert the frame to the hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the colors in the specified range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    nose = 0
    vert = 0
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Find the largest contour
    c = 0
    maxLength = 0
    adj = 0
    adjTop = np.array((0,0))
    adjBottom = np.array((2000,2000))
    for cnt in contours:
        if (cv2.contourArea(cnt) > 500):
            c = cnt
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
        # Draw the contour and the center of the contour on the frame
            saveComp1 = np.array((0,0))
            saveComp2 = np.array((0,0))
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #drawing rectangle around detected objects
            rect_coordinates = []
            x, y, w, h = cv2.boundingRect(c)
            nose = np.degrees(np.arctan(h/w))
            vert = np.degrees(np.arctan((cY - corner[1])/(cX - corner[0])))
            corner = np.array((cX,cY))
            rect_coordinates = np.append(rect_coordinates,[x,y,w,h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0),3)
            pos = np.array((x,y))
            pps = (np.linalg.norm(pos - prevPos))*120
            # print(pps)
            prevPos = pos
            adjust = pps / 2000
    allNose.append(nose)
    allVert.append(vert)
    cv2.putText(frame, 'Nose Angle: {:.2f} Degrees'.format(nose), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame, 'Vertical Launch Angle: {:.2f} Degrees'.format(vert), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    output.write(frame)
    cv2.imshow("Frame", frame)
    # cv2.imshow("mask", mask)
    time.sleep(1)
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(np.average(allNose))
print(np.average(allVert))

# Release the video capture object and close all windows
output.release()
cap.release()
cv2.destroyAllWindows()

