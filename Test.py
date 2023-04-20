print("poop")
import numpy as np 
import cv2
from imutils.video import FPS
# from gi.repository import Gst

# Gst.init(None)
# gstreamer_str = "nvarguscamerasrc sensor-id=1 sensor-mode=2 ! nvv4l2h264enc ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw, format=BGR, enable-max-performace=1 ! appsink drop=1"
gstreamer_str = "nvarguscamerasrc sensor-id=1 sensor-mode=2 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)60/1 ! video/x-raw(memory:NVMM), format=BGR ! nvvidconv ! video/x-raw= ! appsink"
# gstreamer_str = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1 ! nvvidconv ! nvoverlaysink ! appsink drop=1"

# # hsv orange alright?

# gstreamer_str = "nvarguscamerasrc device=/dev/video0 ! video/x-raw,width=1920,height=1080,framerate=60/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw(memory:NVMM), format=I420, width=640, height=360 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink drop=1"
cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture('/home/frolf/video2.mp4')
# output = cv2.VideoWriter('testing1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920,1080))
fps = FPS().start()
gpu_frame = cv2.cuda_GpuMat()
while True:
# Capture the current frameq
    ret, frame = cap.read()
    if not ret:
        break
    gpu_frame.upload(frame)
    gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
    hsv = gpu_hsv.download()
    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow("frame",mask)
    # output.write(frame)
    fps.update()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# output.release()


cap.release()
fps.stop()
# print(fps.elapsed())
print(fps.fps())
cv2.destroyAllWindows()




# from threading import Thread
# import sys
# import cv2
# # import the Queue class from Python 3
# if sys.version_info >= (3, 0):
# 	from queue import Queue
# # otherwise, import the Queue class for Python 2.7
# else:
# 	from Queue import Queue
        
# class FileVideoStream:
# 	def __init__(self, path, queueSize=128):
# 		# initialize the file video stream along with the boolean
# 		# used to indicate if the thread should be stopped or not
# 		self.stream = cv2.VideoCapture(path)
# 		self.stopped = False
# 		# initialize the queue used to store frames read from
# 		# the video file
# 		self.Q = Queue(maxsize=queueSize)
		
#     def start(self):
#         # start a thread to read frames from the file video stream
#         t = Thread(target=self.update, args=())
#         t.daemon = True
#         t.start()
#         return self