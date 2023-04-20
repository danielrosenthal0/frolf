import cv2
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from imutils.video import FPS

np.set_printoptions(threshold=np.inf)
gstreamer_str2 = "nvarguscamerasrc sensor-id=0 exposuretimerange=\"1000000 1000000\" gainrange=\"8 8\" ispdigitalgainrange=\"2 4\" ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap2 = cv2.VideoCapture(gstreamer_str2, cv2.CAP_GSTREAMER)


#white range
lower_white = np.array([0,0,150])
upper_white = np.array([255,30,255])
# # hsv orange alright?
lower_color = np.array([0, 150, 160])
upper_color = np.array([30, 255, 255])

# hsv blue good enough
lower_colorBlue = np.array([0, 0, 0])
upper_colorBlue = np.array([0, 0, 0])

# hsv yellow ish
lower_colorYellow = np.array([30, 70, 130])
upper_colorYellow = np.array([70, 255, 255])

# hsv green 
lower_colorGreen = np.array([60, 60, 70])
upper_colorGreen = np.array([110, 255, 255])


gpu_frame = cv2.cuda_GpuMat()
start = time.time()

i = 1
cornerYellow = np.array((0,0))
fps = FPS().start()

#min area for contours
minArea = 50
maxArea = 4000
while True:
# Capture the current frameq
    ret, frame2 = cap2.read()
    frame2 = frame2[150:420, 0:650]
    if not ret:
        break
    difference = time.time() - start
    cv2.putText(frame2, '{:.2f} FPS'.format(1/difference), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    start = time.time()
    fps.update()

    gpu_frame.upload(frame2)
    gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    hsvSide = gpu_frame.download()

   
    white_mask =  cv2.inRange(hsvSide, lower_white, upper_white)
    maskSide = cv2.inRange(hsvSide, lower_color, upper_color)
    # maskSide = cv2.bitwise_and(maskSide, maskSide, mask=white_mask)
    
    # mask2Side = cv2.inRange(hsvSide, lower_colorBlue, upper_colorBlue)
    # mask3Side = cv2.inRange(hsvSide, lower_colorYellow, upper_colorYellow)
    # mask4Side = cv2.inRange(hsvSide, lower_colorGreen, upper_colorGreen)
    # Use morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
   

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

    all_contours_side, _ = cv2.findContours(maskSide, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
  
    for cnt in all_contours_side:
        area = cv2.contourArea(cnt)
        if area > minArea and area < maxArea:
            c = cnt
            M = cv2.moments(c)
            # if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(frame2, [c], -1, (0, 255, 255), 2)
            cv2.circle(frame2, (cX, cY), 7, (0, 255, 255), -1)
            cv2.putText(frame2, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(c)
            nose = np.degrees(np.arctan(h/w))
            cv2.putText(frame2, 'Nose Angle: {:.2f} Degrees'.format(nose), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow("maskside",maskSide)
    cv2.imshow("Frame2", frame2)
    cv2.imshow("white mask",white_mask)
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
cap2.release()
# cap2.release()
cv2.destroyAllWindows()

