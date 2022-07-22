
import numpy as np
import scipy

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def continuity_fn(drone_x, drone_y, position_prev, count):

    if np.sum(((np.array([drone_x, drone_y])-position_prev)**2)) > 100.:
        count = max((count+1), 3)
        #%% 2 bit counter
        if count<3:
            drone_x = position_prev[0]
            drone_y = position_prev[1]
        
        
    else:
        count = max((count-1), 0)
        
    
    return [drone_x, drone_y, count]