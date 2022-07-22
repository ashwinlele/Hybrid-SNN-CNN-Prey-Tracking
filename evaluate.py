from core import Core
import numpy as np

c = Core()

image_filename = c.current_path + "/DataSets/Drones/testImages/im7.PNG"
image = c.load_image_by_path(image_filename)

print(np.shape(image))
drawing_image = c.get_drawing_image(image)

processed_image, scale = c.pre_process_image(image)

c.set_model(c.get_model())
boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
print(np.shape(image),np.shape(drawing_image),np.shape(processed_image))#,np.shape(boxes), np.shape(scores), np.shape(labels))
print(boxes[0,0,:], scores[0,0], labels[0,0])

detections = c.draw_boxes_in_image(drawing_image, boxes, scores)

c.visualize(drawing_image)
