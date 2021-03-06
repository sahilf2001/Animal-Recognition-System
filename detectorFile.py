import cv2

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath ="frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


def detect():
    cap = cv2.VideoCapture('Bird-3.mp4')
    while(cap.isOpened()):
        success, img = cap.read()
        img = cv2.resize(img, (1240, 680))
        result, objectInfo = getObjects(img, 0.45, 0.2)
        ret, buffer = cv2.imencode('.jpg', result)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if (cv2.waitKey(3) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

