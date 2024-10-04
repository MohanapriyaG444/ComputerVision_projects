from facial_emotion_recognition import EmotionRecognition
import urllib.request
import cv2, imutils
import numpy as np

er = EmotionRecognition(device = 'cpu')
##cam = cv2.VideoCapture(0)

url = "http://192.168.171.203:8080/shot.jpg"

while True:
##    _, frame = cam.read()

    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype = np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width = 500)
    
    
    frame = er.recognise_emotion(frame, return_type = 'BGR')
    cv2.imshow("Emotion Recognition", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
