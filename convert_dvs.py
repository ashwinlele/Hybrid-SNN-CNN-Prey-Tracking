import numpy as np
import matplotlib.pylab as plt
import cv2

v = cv2.VideoCapture('optical_image1.avi')

v.set(1,9)
ret,prev_image = v.read()
prev_image = prev_image.astype(float)
print(np.shape(prev_image))

for k in range(10,21):
    v.set(1,k)
    ret,camera_image = v.read()
    camera_image = camera_image.astype(float)

    print(np.shape(camera_image))
    dvs_image = abs(camera_image[:,:,0] - prev_image[:,:,0])
    
    dvs_image = np.sign(dvs_image - 50)
    dvs_image = (dvs_image + 1)/2
    print(dvs_image)
    plt.imshow(dvs_image)
    plt.show()    
    
    

