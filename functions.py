import csv
import numpy as np
import scipy
#import matcompat
import cv2
import os
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass
import time
from multiprocessing import Process
from core import Core
import argparse
import os
import numpy as np
import tensorflow as tf

from self_cancel_fn2 import *
from bounding_box_fn import *
from continuity_fn import *
from functions import *

def cnn_function(image):
    c = Core()
    drawing_image = c.get_drawing_image(image)
    processed_image, scale = c.pre_process_image(image)
    c.set_model(c.get_model())
    boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
    detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
    drawing_image = cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB)
    return drawing_image

def snn_function(filtered, velocity, depth, window, position_prev, count):
    cancelled = self_cancel_fn2(filtered, velocity, depth)
    [drone_x, drone_y] = bounding_box_fn(cancelled, window)
    [drone_x, drone_y, count] = continuity_fn(drone_x, drone_y, position_prev, count)
    
    return [drone_x, drone_y, count]