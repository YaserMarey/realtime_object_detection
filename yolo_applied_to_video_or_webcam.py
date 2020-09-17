import numpy as np
import cv2

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
output_layers_names = net.getUnconnectedOutLayersNames()
classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#Captuing from Webcam 0, or a video
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("traffic1.mp4")

# Load image
while True:
    ret, img = cap.read()
    # Prepare it
    height, width, _ = img.shape

    # Noramlize, Resize, don't subtract any values, flip GBR to RGB, and don't crop
    # blob formate is the formate that accepted by Deeplearning models
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)

    # Set input
    net.setInput(blob)
    # The output layers are not connected, we would like to connect them
    # that is to set them as THE output layers of the net

    # Now run
    layerOutputs = net.forward(output_layers_names)

    # Now collect the result and visualize them
    boxes = []
    confidences = []
    class_ids = []

    # loop over output and collect results
    for output in layerOutputs:
        for detection in output:
          # Each detection contains 85 parameter, the first four
          # is the location x,y and height and width, then the confidence
          # and the last 80 are proabilites of the object classes
          scores = detection[5:]

          # which class id has the highest score
          class_id = np.argmax(scores)

          # what is the confidence of that class
          confidence = scores[class_id]

          # only if confidence is higher than threshold consider it
          if confidence > 0.5:
              # yolo predictes the center x and y as percentage
              # of the height and width of the image
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)
              # now calculate the upper left corner of the box
              w = int(detection[2] * width)
              h = int(detection[3] * height)
              x = int(center_x - w/2)
              y = int(center_y - h/2)

              boxes.append([x,y, w, h])
              confidences.append(float(confidence))
              class_ids.append(class_id)

    # keep only highest probable box using non maximum suppression
    # 0.5 is the treshold, and 0.4 is the maximum supperssion
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        confidence = str(round(confidences[i],2))
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('', img)

    # Press ESC to leave
    if cv2.waitKey(1) == 27:
        break

# cap.relese()
cv2.destroyAllWindows()
