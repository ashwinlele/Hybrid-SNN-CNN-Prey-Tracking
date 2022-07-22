
import numpy as np
import scipy
import math
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def bounding_box_fn2(V, window):
    height = 1000
    width = 1500
    # Local Variables: c_y, drone_x, drone_y, start_time, ind_x, val_y, V_vert, val_x, V_horz, i, window, ind_y, x, y, filtered, c_x
    # Function calls: plot, floor, max, zeros, bounding_box_fn, size
    #%window = 10;
    
    V_horz = np.zeros((math.floor((width/window)+1), ))
    V_vert = np.zeros((math.floor((height/window)+1), ))
    #print('cancelled',filtered)

    V_vert = np.sum(V,axis = 1) 
    V_horz = np.sum(V,axis = 0)
    '''
    range1 = np.shape(filtered)[1]
    for i in range(1, range1):
        #print('ggg',i)
        x = filtered[0,int(i)-1]
        y = filtered[1,int(i)-1]
        
        c_x = math.floor((x/ window))+1.
        c_y = math.floor((y/ window))+1.
        V_vert[int(c_x-1)] = V_vert[int(c_x-1)]+1.
        V_horz[int(c_y-1)] = V_horz[int(c_y-1)]+1.
    '''
    #print(V_horz)
    ind_x = np.argmax(V_vert)
    ind_y = np.argmax(V_horz)
    
    #print(ind_x,ind_y)
    drone_x = ind_y
    drone_y = ind_x

    
    V[max(int(((drone_y*window)-50))-1,0)-1                                       ,int(max(int(((drone_x*window)-50))-1,0))-1:min(int(((drone_x*window)+50))-1,width-1)] = 1.
    V[min(int(((drone_y*window)+50))-1,height-1)-1                                    ,int(max(int(((drone_x*window)-50))-1,0))-1:min(int(((drone_x*window)+50))-1,width-1)] = 1.
    V[max(int(((drone_y*window)-50))-1,0)-1:min(int(((drone_y*window)+50))-1,height-1),int(max(int(((drone_x*window)-50))-1,0))-1] = 1.
    V[max(int(((drone_y*window)-50))-1,0)-1:min(int(((drone_y*window)+50))-1,height-1),int(min(int(((drone_x*window)+50))-1,width-1))-1] = 1.
    #plt.imshow((255.*V).astype(np.uint8))
    #plt.show()
    #input()
    
    return [drone_x, drone_y]