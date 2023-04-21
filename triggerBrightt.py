import cv2
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from imutils.video import FPS

np.set_printoptions(threshold=np.inf)
gstreamer_str = "nvarguscamerasrc sensor-id=1 exposuretimerange=\"2000000 2000000\" gainrange=\"4 8\" ispdigitalgainrange=\"2 4\" ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
gstreamer_str2 = "nvarguscamerasrc sensor-id=0 exposuretimerange=\"2000000 2000000\" gainrange=\"4 8\" ispdigitalgainrange=\"2 4\" ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
cap2 = cv2.VideoCapture(gstreamer_str2, cv2.CAP_GSTREAMER)

# # hsv orange alright?
lower_color = np.array([5, 70, 175])
upper_color = np.array([30, 255, 255])

# hsv blue good enough
lower_colorBlue = np.array([100, 60, 90])
upper_colorBlue = np.array([180, 255, 255])

# hsv yellow ish
lower_colorYellow = np.array([30, 70, 130])
upper_colorYellow = np.array([70, 255, 255])

# hsv green 
lower_colorGreen = np.array([60, 60, 70])
upper_colorGreen = np.array([110, 255, 255])

# # hsv orange alright?
lower_colorSide = np.array([0, 50, 100])
upper_colorSide = np.array([30, 255, 255])

# output = cv2.VideoWriter('spinRate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1920,1080))
oldAng = 0
allAngles = []
allV = []
allHorz = []
allBank = []
allNose = []
lastV = 0
lastNose = 0
gpu_frame = cv2.cuda_GpuMat()
start = time.time()

i = 1
cornerYellow = np.array((0,0))
fps = FPS().start()

#min area for contours
minArea = 500

minSideArea = 100
maxSideArea = 4000

while True:
# Capture the current frameq
    ret, frame = cap.read()
    ret2, frameSide = cap2.read()
    frame2 = frameSide[150:420, 200:650]
    if not ret:
        break
    difference = time.time() - start
    cv2.putText(frame, '{:.2f} FPS'.format(1/difference), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame2, '{:.2f} FPS'.format(1/difference), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    start = time.time()
    fps.update()
    # if i == 0:
    #     i = 1
    #     continue
    # else:
    #     i = 0
    # Convert the frame to the hsv color space
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gpu_frame.upload(frame)
    gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    hsv = gpu_frame.download()

    gpu_frame.upload(frame2)
    gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    hsvSide = gpu_frame.download()

    # Threshold the frame to get only the colors in the specified range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # mask2 = cv2.inRange(hsv, lower_colorBlue, upper_colorBlue)
    # mask3 = cv2.inRange(hsv, lower_colorYellow, upper_colorYellow)
    # mask4 = cv2.inRange(hsv, lower_colorGreen, upper_colorGreen)
    
    maskSide = cv2.inRange(hsvSide, lower_colorSide, upper_colorSide)
    # mask2Side = cv2.inRange(hsvSide, lower_colorBlue, upper_colorBlue)
    # mask3Side = cv2.inRange(hsvSide, lower_colorYellow, upper_colorYellow)
    # mask4Side = cv2.inRange(hsvSide, lower_colorGreen, upper_colorGreen)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # mask2 = cv2.erode(mask2, kernel, iterations=1)
    # mask2 = cv2.dilate(mask2, kernel, iterations=1)
    # mask3 = cv2.erode(mask3, kernel, iterations=1)
    # mask3 = cv2.dilate(mask3, kernel, iterations=1)
    # mask4 = cv2.erode(mask4, kernel, iterations=1)
    # mask4 = cv2.dilate(mask4, kernel, iterations=1)
    # mask5 = np.bitwise_or(mask,mask2)
    # mask5 = np.bitwise_or(mask5,mask3)
    # mask5 = np.bitwise_or(mask5,mask4)

    # maskSide = cv2.erode(maskSide, kernel, iterations=1)
    # maskSide = cv2.dilate(maskSide, kernel, iterations=1)
    # mask2Side = cv2.erode(mask2Side, kernel, iterations=1)
    # mask2Side = cv2.dilate(mask2Side, kernel, iterations=1)
    # mask3Side = cv2.erode(mask3Side, kernel, iterations=1)
    # mask3Side = cv2.dilate(mask3Side, kernel, iterations=1)
    # mask4Side = cv2.erode(mask4Side, kernel, iterations=1)
    # mask4Side = cv2.dilate(mask4Side, kernel, iterations=1)
    # mask5Side = np.bitwise_or(maskSide,mask2Side)
    # mask5Side = np.bitwise_or(mask5Side,mask3Side)
    # mask5Side = np.bitwise_or(mask5Side,mask4Side)

    # Find the contours in the mask
    # blue_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # orange_contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    all_contours_side, _ = cv2.findContours(maskSide, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    doCon = False
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            doCon = True
            break
    if doCon is False:
        if np.any(allV):
            lastNose = np.median(allNose)
            allNose = []
            lastV = allV[-1]
            allV = []
        cv2.putText(frame, 'Nose Angle: {:.2f} Degrees'.format(lastNose), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Velocity: {:.2f} MPH'.format(lastV), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Velocity", frame)
        cv2.imshow("Side", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # for cnt in blue_contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 5000:
    #         c = cnt
    #         M = cv2.moments(c)
    #         # if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         centerOrange = np.array((cX,cY))
    #         cv2.drawContours(frame, [c], -1, (51, 87, 255), 2)
    #         cv2.circle(frame, (cX, cY), 7, (51, 87, 255), -1)
    #         cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 87, 255), 2)

    # for cnt in orange_contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 5000:
    #         c = cnt
    #         M = cv2.moments(c)
    #         # if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         centerBlue = np.array((cX,cY))
    #         cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
    #         cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
    #         cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    mph = 0
    horz = 0
    bank = 0
    adjTop = np.array((0,0))
    adjBottom = np.array((100000,100000))
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
            cv2.circle(frame, (cX, cY), 7, (0, 255, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            (x,y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 255), 2)

            longest = 1.75 * radius
            cv2.putText(frame, "{:.2f}".format(longest), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # longest = 112
            # for j in range(len(c)):
            #     comp = np.array((c[j][0][0], c[j][0][1]))
            #     longest = max(longest, np.linalg.norm(comp-np.array((cX,cY))))
            #     if comp[1] > adjTop[1]:
            #         adjTop = comp
            #     elif comp[1] < adjBottom[1]:
            #         adjBottom = comp
            x, y, w, h = cv2.boundingRect(c)
            # longest = w
            pps = np.linalg.norm(cornerYellow - np.array((cX,cY)))*(20)
            cmpp = 21.59/longest
            cmps = pps*cmpp
            mpcm = 1/160900
            mph = cmps * mpcm * 60 * 60
            cornerYellow = np.array((cX,cY))

    for cnt in all_contours_side:
        area = cv2.contourArea(cnt)
        if area > minSideArea and area < maxSideArea:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(frame2, [c], -1, (0, 255, 255), 2)
            cv2.circle(frame2, (cX, cY), 7, (0, 255, 255), -1)
            cv2.putText(frame2, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            rows,cols = frame2.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2, 0,0.01,0.01)
            left = int((-x*vy/vx)+y)
            right = int(((x)*vy/vx)+y)
            # cv2.line(frame2,((cols-1),int(right)),(0,int(left)),(0,255,0),2)
            try:
                nose = -np.degrees(np.arctan((left-right)/(cols-1)))
            except:
                nose = 0
            cv2.putText(frame2, 'Nose Angle: {:.2f} Degrees'.format(nose), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            allNose.append(nose)
            # for j in range(len(c)):
            #     comp = np.array((c[j][0][0], c[j][0][1]))
            #     longest = max(longest, np.linalg.norm(comp-np.array((cX,cY))))
            #     if comp[1] > adjTop[1]:
            #         adjTop = comp
            #     elif comp[1] < adjBottom[1]:
            #         adjBottom = comp
            # x, y, w, h = cv2.boundingRect(c)
            # # longest = w
            # pps = np.linalg.norm(cornerYellow - np.array((x,y)))*(1/difference)
            # cmpp = 21.59/longest
            # cmps = pps*cmpp
            # mpcm = 1/160900
            # mph = cmps * mpcm * 60 * 60
            # cornerYellow = np.array((x,y))

    # if mph > 0:
    allV.append(mph)
    # print(mph)
    
    cv2.putText(frame, 'Velocity: {:.2f} MPH'.format(mph), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'Nose Angle: {:.2f} Degrees'.format(nose), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
       
    # cv2.imshow("Frame2", frame2)
    cv2.imshow("Velocity", frame)
    cv2.imshow("Side", frame2)
    cv2.imshow("mask5", maskSide)
    # cv2.imshow("test",mask5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(np.average(allV)) 
fps.stop()
print(fps.fps())

# print(allV)
# print(np.average(allAngles[~np.isnan(allAngles)])) 


# Release the video capture object and close all windows
# output.release()
cap.release()
# cap2.release()
cv2.destroyAllWindows()

