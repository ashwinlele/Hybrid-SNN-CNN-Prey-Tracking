# Author: aqeelanwar
# Created: 19 March,2020, 7:56 AM
# Email: aqeel.anwar@gatech.edu

import cv2
import numpy as np
from skimage.util import random_noise

video_file_path = 'dvs.mp4'
cap = cv2.VideoCapture(video_file_path)
thresh1 = 0.07

w = 600
if (cap.isOpened()== False):
  print("Error opening video stream or file")

first_frame = True
while(cap.isOpened()):
  ret, frame1 = cap.read()
  w = frame1.shape
  frame1 = cv2.resize(frame1, (int(w[1]/2),int(w[0]/2)))
  frame1 = frame1[0: 500, 50:800]
  frame_rgb = cv2.normalize(frame1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  if ret == True:
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame = np.uint8(np.log1p(frame1))
    frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if first_frame:
      display_frame = frame
      first_frame = False
    else:
      display_frame = frame-last_frame
    last_frame = frame
    #thresh1 = random_noise(thresh1, mode='s&p', amount=0.005)
    display_frame = random_noise(display_frame, mode='s&p', amount=0.01)

    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    d = cv2.hconcat([frame_rgb, display_frame])

    cv2.imshow('dvs_output', d)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else:
    break

cap.release()
cv2.destroyAllWindows()