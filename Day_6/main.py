
## reading frame from camera
## pre-processing image
## finding contours
## drawing minimum enclosing circle
## finding centre of contour area (moments)
## drawing circle & dot
## showing directions based on radius & position

import imutils
import cv2

## blue ==> low=(98,67,57), high=(130, 255, 127)
## red  ==> low=(0,75,103), high=(10, 255, 255)
## my face ==> Low = [0,19,78], High= [57,150,223]
lower = (0,19,78)
upper = (57,150,223)

cam = cv2.VideoCapture(0) # cam initiate

while True:

    _,frame = cam.read()    # reading frame from camera

    #preprocessing
    frame = imutils.resize(frame, width=1000)   #resize
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #mask the color
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)

    # finding contours
    contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # drawing minimum enclosing circle
    if len(contour)>0:
        c = max(contour, key = cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)

        # finding centre of contour area (moments)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # drawing circle & dot
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


            # showing directions based on radius & position
            if(radius > 250):
               print("Stop")
            else:
                if(center[0]<150):
                    print("Right")
                elif(center[0]>450):
                    print("Left")
                elif(radius<250):
                    print("Front")
                else:
                    print("stop")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
        


