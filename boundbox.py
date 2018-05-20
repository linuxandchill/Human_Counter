import numpy as np
import cv2 
import imutils

#open file
cap = cv2.VideoCapture("test2.mp4")

#create background subtractor
bgsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

opening_filter = np.ones((3,3), np.uint8)
closing_filter = np.ones((11,11), np.uint8)
threshold_area = 1000

while(True):
    #read one frame
    ret, frame = cap.read()

    if not ret:
        break

    #resize
    frame = imutils.resize(frame, width = 800)

    #make gray and blur
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    #use subtractor
    fg_mask = bgsub.apply(frame)
    imBin = cv2.threshold(fg_mask,200,255,cv2.THRESH_BINARY)[1]
    #eliminate some noise by eroding-> dilating
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, opening_filter)
    #dilate -> erode to merge white
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_filter)


    ####CONTOURS#######
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0,255,0), 3, 8)
        area = cv2.contourArea(contour)
        print(area)

        if area > threshold_area:
            moment = cv2.moments(contour)
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])

            (x,y,wid,hei) = cv2.boundingRect(contour)
            cv2.circle(frame,(cx,cy), 10, (255,0,255), -1)            
            cv2.rectangle(frame, (x,y), (x + wid, y + hei), (255, 0 , 255), 4)

    cv2.imshow("Frame", frame)  

    #kill on keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
