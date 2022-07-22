from core import Core
import argparse
import os
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from matplotlib import pyplot as plt
from PIL import Image

import scipy
import cv2
import numpy as np

cap = cv2.VideoCapture('dense_close_fast.mp4')

cap.set(1,300);

ret, frame1 = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('labelled1.avi',fourcc,30,(1728,1080),1) #160,128),0)

c = Core()

for p in range(300,900):
    print(p)
    cap.set(1,p) # Where frame_no is the frame you want
    ret, image = cap.read() # Read the frame

    print(type(image))
    drawing_image = c.get_drawing_image(image)
    processed_image, scale = c.pre_process_image(image)
    c.set_model(c.get_model())
    boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
    detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
    #c.visualize(drawing_image)
    print(np.shape(drawing_image))
    drawing_image = cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB)
    video.write(drawing_image) #.astype('int32')
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next
        
cap.release()
cv2.destroyAllWindows()
video.release()


