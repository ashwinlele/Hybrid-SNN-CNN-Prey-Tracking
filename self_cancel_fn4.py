import time
    
import numpy as np
import math
import random

def self_cancel_fn4(velocity, depth, dvs_image, position_prev, count):

    start_time = time.time()
    
    prev_time = 0
    
    delta_v = 1
    height = 480
    width = 640
    window = 10
    
    mult_v = 1. #3.2
    mult_depth = 0.2 #0.8
    [x,y] = np.nonzero(dvs_image)
   
    vx = velocity[:,0:width] #velocity[:,960:1920]
    vy = velocity[:,width:] #velocity[:,0:960]
    span = 12 #max(int(np.max(vx)),int(np.max(vy)))
    #print('span1',int(np.max(vx)),int(np.max(vy)))
    dvs_vert = np.zeros((height+span, width))
    dvs_horz = np.zeros((height, width+span))
    
    for i in range(0,span): #int(np.max(vx))):
        dvs_vert[i:height+i,:] = np.add(dvs_vert[i:height+i,:],dvs_image)
    for i in range(0,span): #int(np.max(vy))):
        dvs_horz[:,i:width+i] = np.add(dvs_horz[:,i:width+i] , dvs_image)
    #print('span2',np.shape(vx),np.shape(vy))
    V1 = np.sign((dvs_vert[0:height,:]  - np.max(vx)*(1-0/2+0*random.random())  - np.exp(depth*0.4) )) #- np.max(vx) + 1)) #(2-np.max(vx))*np.ones((height,width)))*dvs_vert[0:height,:])
    V2 = np.sign((dvs_horz[:, 0:width]  - np.max(vy)*(1-0/2+0*random.random())  - np.exp(depth*0.4) )) #- np.max(vy) + 1)) #(2-np.max(vy))*np.ones((height,width)))*dvs_horz[:,0:width])
    print('velocity',np.max(vx),np.max(vy))
    V1 = np.where(V1 < 0.1, 0, 1) 
    V2 = np.where(V2 < 0.1, 0, 1) 
    V = np.logical_and(V1, V2)
    #print('span3',int(np.max(vx)),int(np.max(vy)))

    V_vert = np.sum(V,axis = 1) 
    V_horz = np.sum(V,axis = 0)
    
    #print(" #### SNN 2 %s seconds ---" % (time.time() - start_time))  
    
    V_horz1 = np.zeros((math.floor((width/window)+1), ))
    V_vert1 = np.zeros((math.floor((height/window)+1), ))
    #print('span4',int(np.max(vx)),int(np.max(vy)))

    for i in range(0,math.floor(height/window)):
        V_vert1[i] = np.sum(V_vert[i*window:(i+1)*window])
    for i in range(0,math.floor(width/window)):
        V_horz1[i] = np.sum(V_horz[i*window:(i+1)*window])
    
    ind_x = np.argmax(V_vert1)
    ind_y = np.argmax(V_horz1)
    #print('span5',int(np.max(vx)),int(np.max(vy)))

    drone_x = ind_y
    drone_y = ind_x

    if np.sum(((np.array([drone_x, drone_y])-position_prev/window)**2)) > 25.:
        count = max((count+1), 3)
        #%% 2 bit counter
        if count<3:
            drone_x = position_prev[0]
            drone_y = position_prev[1]
    else:
        count = max((count-1), 0)
    #print('span6',int(np.max(vx)),int(np.max(vy)))

    V[max(int(((drone_y*window)-50))-1,0)-1                                       ,int(max(int(((drone_x*window)-50))-1,0))-1:min(int(((drone_x*window)+50))-1,width-1)] = 1.
    V[min(int(((drone_y*window)+50))-1,height-1)-1                                    ,int(max(int(((drone_x*window)-50))-1,0))-1:min(int(((drone_x*window)+50))-1,width-1)] = 1.
    V[max(int(((drone_y*window)-50))-1,0)-1:min(int(((drone_y*window)+50))-1,height-1),int(max(int(((drone_x*window)-50))-1,0))-1] = 1.
    V[max(int(((drone_y*window)-50))-1,0)-1:min(int(((drone_y*window)+50))-1,height-1),int(min(int(((drone_x*window)+50))-1,width-1))-1] = 1.

    V3 = np.append(dvs_image,V,axis = 0)
    V4 = np.append(((dvs_vert[0:height,:]/np.max(vx)).astype(float)),((dvs_horz[:,0:width]/np.max(vy)).astype(float)),axis = 0)
    V3 = np.append(V3,V4,axis = 1)
    V4 = np.append(V1,V2,axis = 0) #
    V3 = np.append(V3,V4,axis = 1)
    
    return [V, drone_x*window, drone_y*window, V3, count]
