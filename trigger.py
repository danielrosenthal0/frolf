import cv2
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt


np.set_printoptions(threshold=np.inf)
gstreamer_str = "nvarguscamerasrc sensor-id=1 exposuretimerange=\"2000 2000\" ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
gstreamer_str2 = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
cap2 = cv2.VideoCapture(gstreamer_str2, cv2.CAP_GSTREAMER)

# # hsv orange alright?
lower_color = np.array([0, 70, 150])
upper_color = np.array([30, 255, 255])

# hsv blue good enough
lower_colorBlue = np.array([100, 60, 100])
upper_colorBlue = np.array([180, 255, 255])

# hsv yellow ish
lower_colorYellow = np.array([30, 70, 150])
upper_colorYellow = np.array([70, 255, 255])

# hsv green 
lower_colorGreen = np.array([60, 70, 70])
upper_colorGreen = np.array([110, 255, 255])

# output = cv2.VideoWriter('spinRate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1920,1080))
oldAng = 0
allAngles = []
allV = []
allHorz = []
allBank = []
lastV = 0
gpu_frame = cv2.cuda_GpuMat()
start = time.time()

i = 1
cornerYellow = np.array((0,0))
while True:
# Capture the current frameq
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret:
        break
    difference = time.time() - start
    start = time.time()
    # if i == 0:
    #     i = 1
    #     continue
    # else:
    #     i = 0

    centerOrange = np.array((0,0))
    centerBlue = np.array((0,0))

    # Convert the frame to the hsv color space
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gpu_frame.upload(frame)
    gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    hsv = gpu_frame.download()

    # Threshold the frame to get only the colors in the specified range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask2 = cv2.inRange(hsv, lower_colorBlue, upper_colorBlue)
    mask3 = cv2.inRange(hsv, lower_colorYellow, upper_colorYellow)
    mask4 = cv2.inRange(hsv, lower_colorGreen, upper_colorGreen)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask2 = cv2.erode(mask2, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask3 = cv2.erode(mask3, kernel, iterations=1)
    mask3 = cv2.dilate(mask3, kernel, iterations=1)
    mask4 = cv2.erode(mask4, kernel, iterations=1)
    mask4 = cv2.dilate(mask4, kernel, iterations=1)
    mask5 = np.bitwise_or(mask,mask2)
    mask5 = np.bitwise_or(mask5,mask3)
    mask5 = np.bitwise_or(mask5,mask4)

    if 255 not in mask5:
        if np.any(allV):
            lastV = np.average(allV)
            allV = []
        cv2.putText(frame, 'Velocity: {:.2f} MPH'.format(lastV), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.imshow("VScreen", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours3, _ = cv2.findContours(mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    
    for cnt in contours:
        if len(cnt) > 175 and len(cnt) < 1000:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centerOrange = np.array((cX,cY))
            cv2.drawContours(frame, [c], -1, (51, 87, 255), 2)
            cv2.circle(frame, (cX, cY), 7, (51, 87, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 87, 255), 2)

    for cnt in contours2:
        if len(cnt) > 175 and len(cnt) < 1000:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centerBlue = np.array((cX,cY))
            cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    mph = 0
    horz = 0
    bank = 0
    adjTop = np.array((0,0))
    adjBottom = np.array((100000,100000))
    for cnt in contours3:
        if len(cnt) > 175 and len(cnt) < 1000:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
            cv2.circle(frame, (cX, cY), 7, (0, 255, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            longest = 0
            for j in range(len(c)):
                comp = np.array((c[j][0][0], c[j][0][1]))
                longest = max(longest, np.linalg.norm(comp-np.array((cX,cY))))
                if comp[1] > adjTop[1]:
                    adjTop = comp
                elif comp[1] < adjBottom[1]:
                    adjBottom = comp
            longest = longest * 2
            x, y, w, h = cv2.boundingRect(c)
            pps = np.linalg.norm(cornerYellow - np.array((x,y)))*(1/difference)
            cmpp = 21/longest
            cmps = pps*cmpp
            mpcm = 1/160900
            mph = cmps * mpcm * 60 * 60
            cornerYellow = np.array((x,y))

    
    # if mph > 0:
    allV.append(mph)
    cv2.putText(frame, 'Velocity: {:.2f} MPH'.format(mph), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.imshow("Frame2", frame2)
    cv2.imshow("Frame", frame)

    # cv2.imshow("test",mask5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(np.average(allV)) 


# print(allV)
# print(np.average(allAngles[~np.isnan(allAngles)])) 


# Release the video capture object and close all windows
# output.release()
cap.release()
cap2.release()
cv2.destroyAllWindows()

