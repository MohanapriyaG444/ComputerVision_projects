import cv2

## loading the haarcascade_frontalface algorithm
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

## initializing the camera
cam = cv2.VideoCapture(0)

while True:

    _,frame = cam.read()    ##reading frame from camera
    gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    ##converting the image in the frame to grayscale

    ##obtaining the face coordinates using alg
    face = haar_cascade.detectMultiScale(gFrame, 1.3, 4)

    for x, y, w, h in face:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    ## display
    cv2.imshow("Face detection", frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
