import cv2
import numpy as np

# brew install opencv@2 / follow step from brew / do 'make' in darknet
# pip install opencv-python==3.4.8.29

# tiny
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# 608
# net = cv2.dnn.readNet('yolov3-608.weights', 'yolov3-608.cfg')

# spp
# net = cv2.dnn.readNet('yolov3-spp.weights', 'yolov3-spp.cfg')

# own model
# net = cv2.dnn.readNet('yolov3-voc_30000.weights', 'yolov3-voc.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('test5.mp4')
while True:
    _, img = cap.read()

    try:
        img = cv2.resize(img, (960, 540))

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        # for b in blob:
        #     for n, img_blob in enumerate(b):
        #         cv2.imshow(str(n), img_blob)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        print(len(boxes))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        # print(indexes.flatten())

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3), )

        if len(indexes) != 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                if class_ids[i] in [0, 3, 58]:  # three classes
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                    cv2.imshow('Image', img)

        key = cv2.waitKey(1)

    except Exception as e:
        # print(e.args, flush=True)
        print("Finish !!", flush=True)
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
