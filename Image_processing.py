import cv2
import numpy as np
# import time
# import pandas as pd
# from matplotlib import pyplot as plt
# from PIL import Image as im

np.set_printoptions(threshold=np.inf)

gstreamer_str = "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! nvv4l2h264enc ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw, format=BGR, enable-max-performace=1 ! appsink drop=1"

cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)


# # hsv orange alright?
lower_color = np.array([0, 70, 180])
upper_color = np.array([30, 255, 255])

# hsv blue good enough
lower_colorBlue = np.array([100, 60, 120])
upper_colorBlue = np.array([180, 255, 255])

# hsv yellow ish
lower_colorYellow = np.array([30, 70, 170])
upper_colorYellow = np.array([70, 255, 255])

output = cv2.VideoWriter('spinRate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920,1080))
oldAng = 0
allAngles = []
allV = []
allHorz = []
allBank = []
i = 1
cornerYellow = np.array((0,0))
while True:
# Capture the current frameq
    ret, frame = cap.read()
    if not ret:
        break

    # if i == 0:
    #     i = 1
    #     continue
    # else:
    #     i = 0

    centerOrange = np.array((0,0))
    centerBlue = np.array((0,0))

    # Convert the frame to the hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the colors in the specified range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask2 = cv2.inRange(hsv2, lower_colorBlue, upper_colorBlue)
    mask3 = cv2.inRange(hsv3, lower_colorYellow, upper_colorYellow)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask2 = cv2.erode(mask2, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask3 = cv2.erode(mask3, kernel, iterations=1)
    mask3 = cv2.dilate(mask3, kernel, iterations=1)

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    
    for cnt in contours:
        if len(cnt) > 200:
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
        if len(cnt) > 200:
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
        if len(cnt) > 200:
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
            pps = np.linalg.norm(cornerYellow - np.array((x,y)))*60
            horz = np.degrees(np.arctan((y - cornerYellow[1])/(x - cornerYellow[0])))
            bank = np.degrees(np.arccos(np.linalg.norm(adjTop - adjBottom)/longest))
            cmpp = 21/longest
            cmps = pps*cmpp
            mpcm = 1/160900
            mph = cmps * mpcm * 60 * 60
            cornerYellow = np.array((x,y))


    adj = centerBlue[0] - centerOrange[0]
    opp = centerBlue[1] - centerOrange[1]
    ang = np.arctan(adj/opp)
    ang = np.degrees(ang)

    if centerOrange[1] == 0 or centerBlue[1] == 0:
        rps = 0
    elif centerOrange[0] > 1900 or centerBlue[0] > 1900:
        rps = 0
    elif centerOrange[0] < 150 or centerBlue[0] < 150:
        rps = 0
    elif ang < 0 and oldAng > 0:
        change = (ang + 180) - oldAng
        rps = abs((change)*60/360)
        # cv2.line(frame, centerOrange, centerBlue, (255, 255, 255), thickness=3, lineType=8)
        # cv2.putText(frame, '{:.2f}'.format(ang), (centerOrange[0] - 20, centerOrange[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        change = ang - oldAng
        rps = abs((change)*60/360)
        # cv2.line(frame, centerOrange, centerBlue, (255, 255, 255), thickness=3, lineType=8)
        # cv2.putText(frame, '{:.2f}'.format(ang), (centerOrange[0] - 20, centerOrange[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # print(oldAng)
    # print(ang)
    # print(rps)
    print("")
    oldAng = ang
    if rps > 0:
        allAngles.append(rps)  
    
    # if mph > 0:
    allV.append(mph)
    # if abs(horz) != 0:
    allHorz.append(horz)
    # if bank > 0:
    allBank.append(bank)
    cv2.putText(frame, 'Spin Rate: {:.2f} R/S'.format(rps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame, 'Velocity: {:.2f} MPH'.format(mph), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame, 'Horizontal Launch Angle: {:.2f} Degrees'.format(horz), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame, 'Bank: {:.2f} Degrees'.format(bank), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    # output.write(frame)
    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(np.average(allAngles)) 
print(np.average(allV)) 
print(np.average(allHorz)) 
print(np.average(allBank)) 

# print(allV)
# print(np.average(allAngles[~np.isnan(allAngles)])) 


# Release the video capture object and close all windows
# output.release()
cap.release()
cv2.destroyAllWindows()