import numpy as np 
import cv2
from imutils.video import FPS

# gstreamer_str = "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! video/x-raw(memory:NVMM), width=(int)720, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
# gstreamer_str = "nvarguscamerasrc sensor-id=1 sensor-mode=2 ! video/x-raw(memory:NVMM), width=(int)720, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! cudaupload ! videoconvert ! nvegltransform ! nveglglessink -e"
# gstreamer_str = "nvarguscamerasrc sensor-id=1 sensor-mode=2 ! video/x-raw(memory:NVMM) ! nvvidconv ! nvivafilter cuda-process=true customer-lib-name=lib-gst-custom-opencv_cudaprocess.s ! video/x-raw(memory:NVMM), format=BGR ! appsink"
# gstreamer_str2 = "nvarguscamerasrc sensor-id=1 sensor-mode=2 ! video/x-raw(memory:NVMM), width=(int)720, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
# # hsv orange alright?


gstreamer_str2 = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(gstreamer_str2, cv2.CAP_GSTREAMER)
# cap2 = cv2.VideoCapture(gstreamer_str2, cv2.CAP_GSTREAMER)
gpu_frame = cv2.cuda_GpuMat()
fps = FPS().start()
# gpu_frame = cv2.cuda_GpuMat()
while True:
    ret, frame = cap.read()
    gpu_frame.upload(frame)
    gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    lower_color = cv2.cuda_GpuMat(gpu_hsv.size(), cv2.CV_8UC3, (0, 70, 150))
    upper_color =  cv2.cuda_GpuMat(gpu_hsv.size(), cv2.CV_8UC3, (30, 255, 255)) 
    # Threshold the frame to get only the colors in the specified range
    gpu_mask = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC1)
    # cv2.cuda.createContinuous(480, 720, cv2.CV_8UC1,gpu_mask)
    # cv2.cuda.inRange(gpu_hsv, lower_color, upper_color, gpu_mask)
    cv2.cuda.inRange(gpu_hsv, tuple(lower_color.download().reshape(1, 1, 3)[0][0]), tuple(upper_color.download().reshape(1, 1, 3)[0][0]), gpu_mask)

    # ret2, frame2 = cap2.read()
    mask = gpu_mask.download()
    fps.update()
    cv2.imshow("frame",mask)
    # cv2.imshow("frame2",frame2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
fps.stop()
cap.release()

cv2.destroyAllWindows()
print(fps.fps())