import cv2
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image as im
np.set_printoptions(threshold=np.inf)

# Initialize the video capture object
cap = cv2.VideoCapture('/Users/jackconnelly/Documents/School/Fall_2022/EGR_314/OpenCV/Kendall_5_trimmed_2.MP4')
# cap = cv2.VideoCapture(0)


# Define the range of colors for the disc
# yellow range
# lower_color = np.array([196, 208, 25])
# upper_color = np.array([236, 248, 45])

# orange range
# lower_color = np.array([160, 0, 30])
# upper_color = np.array([255, 255, 100])

# hsv?
lower_color = np.array([0, 50, 170])
upper_color = np.array([255, 255, 255])

# ######################################################################################################################################
# black rgb lower upper
lower_colorb = np.array([0, 0, 0])
upper_colorb = np.array([255, 255, 100])
# ######################################################################################################################################

#object detection parameters
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold= 90)


maxLength = 0
loc = ""
centerX = []
centerY = []
area = []
while True:
    
    # Capture the current frameq
    ret, frame = cap.read()
    frame2 = frame
    if not ret:
        break
    # Convert the frame to the hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rgb color space for second mask
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the colors in the specified range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x = np.array([])
    y = np.array([])

    # Find the largest contour
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area.append(cv2.contourArea(c))
        # Find the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # saving all locations of center of contours for calculations 
            centerX.append(cX)
            centerY.append(cY)
        else:
            cX, cY = 0, 0
        # Draw the contour and the center of the contour on the frame
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(len(contours))
        coord = c.ravel()
        i = 0
        # getting greatest distance between every point of contour to get true diameter
        for word in coord:
            if i % 2 == 1:
                i = i + 1
                continue
            comp = np.array((coord[i], coord[i+1]))
            x = np.append(x,coord[i])
            y = np.append(y,coord[i+1])
            j = 0
            for newword in coord:
                if j % 2 == 1:
                    j = j + 1
                    continue
                newComp = np.array((coord[j], coord[j+1]))
                maxLength = max(maxLength, abs(np.linalg.norm(comp - newComp)))
                j = j + 1
            i = i + 1
    # extracting data to local coordinate system whenever contour is detected
    try:
        x = np.subtract(x, np.amin(x) - 1)
        y = np.subtract(y, np.amin(y) - 1)
        xmax = int(np.amax(x) + 1)  
        ymax = int(np.amax(y) + 1)
        loc = np.zeros((xmax, ymax), dtype=np.int16)
        i = 0
        for num in x:
            loc[int(x[i]),int(y[i])] = 255
            i = i + 1
    except:
        print("woops")

    # mask2 = frame2[100:720, 500:1250]
    mask2  = cv2.inRange(rgb, lower_colorb, upper_colorb)
    mask2 = cv2.erode(mask2, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)

    # motion tracking
    # mask2 = object_detector.apply(mask2)

    contours2, h = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours2) > 0:
        c = max(contours2, key=cv2.contourArea)
        # Find the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        # Draw the contour and the center of the contour on the frame
        cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    

    # Show the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("mask2", mask2)
    # time.sleep(.3)
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

print(maxLength)

# immer = im.fromarray(loc, "BW")
# immer = im.fromarray(loc.astype('uint8'))
# immer.save("image.png")




# logic for velocity and horizontal launch angle calculations
inV = []
lastVal = np.array([])
i = 0
area = np.divide(346.5,area)
area = np.sqrt(area)
mul = 120 * 0.0000062137 * 60 * 60
area = np.multiply(area, mul)
coef = ((21/maxLength)*120)* 0.0000062137 * 60 * 60
horz = []
for poop in centerX:
    thisVal = np.array((centerX[i], centerY[i]))
    if len(lastVal) == 2:
        inV.append(abs(np.linalg.norm(thisVal - lastVal)))
        horz.append(np.arctan((thisVal[1]-lastVal[1])/(thisVal[0]-lastVal[0])))
    lastVal = thisVal
    i = i + 1
horz = np.degrees(horz)
print(horz)
inV = np.multiply(inV, coef)
# print(inV)

