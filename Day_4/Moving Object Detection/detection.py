import cv2
import time ##delay
import imutils ##resize

##initialize the camera
cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500

while True:
    _, frame = cam.read() #read the frame from camera
    text = "normal"
    frame = imutils.resize(frame, width = 1000) ##resizing the frame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    ##color to grayscale
    blurredFrame = cv2.GaussianBlur(grayFrame,(21,21),0)    ## smoothening

    ## capturing the first frame(i.e., a stable frame to detect if any object is moving)
    if firstFrame is None:
        firstFrame = blurredFrame
        continue

    frameDiff = cv2.absdiff(firstFrame, blurredFrame) ## finding difference
    bFrame = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]   ##applying threshold to get binary frame
    bFrame = cv2.dilate(bFrame, None, iterations=2)   ## Dilation to fill in gaps
    contour = cv2.findContours(bFrame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    
    for c in contour:

        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)
        text = "Alert!!! moving object detected...."
    print(text)
    cv2.putText(frame,text,(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("Security Camera....", frame)
    key = cv2.waitKey(1) &0xFF

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
    
        
