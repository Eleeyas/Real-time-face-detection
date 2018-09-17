import numpy as np
import cv2
import dlib

#MOUTH_POINT = 48 TO 68
#RIGHT_BROW_POINT = 17 TO 22
#LEFT_BROW_POINT = 22 TO 27
#RIGHT_EYE_POINT = 36 TO 42
#LEFT_EYE_POINT = 42 TO 48
#NOSE_POINT = 27 TO 35
#JAW_POINT = 0 TO 17
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype) # 68 is all array and 2 is 2-tuple of (x,y)-coordinates
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predector = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)

#print("")
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_strat, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        rects = detector(gray, 0)

        for rect in rects:
            shape = predector(gray, rect) 
            shape = shape_to_np(shape) #I/P x, y
        
            print(shape)    
            for (xx, yy) in shape:
                cv2.circle(frame, (xx, yy), 1, (0, 255, 0), 1) #O/P

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255,0,0) 
        stroke = 1
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y),color,stroke)
    
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


