# ----Deteksi Objek
import cv2


frozen_model = 'frozen_model.pb'
conf_file = 'conf.pbtxt'

model = cv2.dnn_DetectionModel(frozen_model, conf_file)

classLabels = []
label_file = 'Labels.txt'
with open(label_file, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# img = cv2.imread('gambar.jpg')
cap = cv2.VideoCapture('potrait.mp4')
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("gagal membuka webcam")

while True:
    ret, frame = cap.read()
    frames = cv2.resize(frame, (540, 960))
    classIndex, confidence, bbox = model.detect(frames, confThreshold=0.6)

    if(len(classIndex) != 0):
        for classId, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classId <= 80:
                # print(classId)
                cv2.rectangle(frames, boxes, (255, 0, 0), 2)
                cv2.putText(frames, classLabels[classId-1], (boxes[0],
                                                             boxes[1]-4), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.putText(frames, "ICARE-EYE (SMART VISION)", (10, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frames, "DEMO", (400, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("ICARE-EYE (SMART VISION) DEMO", frames)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
