import cv2, numpy, os

haar_file = "haarcascade_frontalface_default.xml"
dataset = "dataset"

print("Training in progress...")

(images, labels, names, ids) = ([], [], {}, 0)

for(subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[ids] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath+'/'+filename
            label = ids
            images.append(cv2.imread(path,0))
            labels.append(int(label))

        ids += 1

(width, height) = (130,100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

#print(images, labels)
model = cv2.face.LBPHFaceRecognizer_create()
##model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

cascade_algo = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)
count = 0

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_algo.detectMultiScale(gray, 1.3, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize) ##output
        cv2.rectangle(img, (x, y),(x+w, y+h), (0,255,0),2)
        if prediction[1]<800:
            cv2.putText(img,'%s - %.0f'%(names[prediction[0]],prediction[1]), (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (51, 200, 255))
            print(names[prediction[0]])

            count = 0

        else:
            count += 1
            cv2.putText(img, "Unknown", (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            if count > 100 :
                print("Unknown person")
                cv2.imwrite("unknown.jpg", img)
                count = 0

    cv2. imshow("Face Recognition", img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

        
