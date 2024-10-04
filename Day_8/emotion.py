from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device = 'cpu')
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = er.recognise_emotion(frame, return_type = 'BGR')
    cv2.imshow("Emotion Recognition", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
