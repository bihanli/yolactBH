import cv2
import numpy as np
from PIL import Image
import os
f_path ="../TUD--2020-05-11--03-00-57--17--fcamera.hevc"
file_path ="../Comma_TUD.mp4"

Comma_capture = cv2.VideoCapture(f_path) 

fps = Comma_capture.get(cv2.CAP_PROP_FPS)
Comma_writer = cv2.VideoWriter(file_path,
			       cv2.VideoWriter_fourcc(*'MP4V'),
			       fps,
			       (1164,874))

success, frame = Comma_capture.read()
print(success)
while success and not cv2.waitKey(1) == 27:
    # 
    Comma_writer.write(frame) # 
    #cv2.imshow("video", video)
    success, frame = Comma_capture.read()

#
Comma_capture.release()
Comma_writer.release()

if not os.path.exists("../output_video"):
	os.mkdir ("../output_video")
#output_path = "../output_video/Comma_TUN_YOLACT.mp4"
