# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:43:32 2019

@author: ASUS
"""
import cv2

#Getting the cascades to train the model
#Getting cascade of faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Getting cascade of eyes 
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')


#detect function will take coloured and gray image and it will return the same image with the detector rectangles
def detect(frame,gray):
    
    #detectMultiScale method will find the place on the frame and it will return cordinates (x,y) , width , height.It can be multiple face on single frame also
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    #now for loop will iterate throught every found face, and draw rectangle around it
    for (x,y,w,h) in faces:
        
        #rectangle function will draw box arounnd the face found in frame. it will take cordinates of the founded image,and color , width of the box
        cv2.rectangle(frame,(x,y) , (x+w ,y+h) ,(255,0,0) , 2)
        
        #Eye can be found inside the face so selectiong inside part of face to detect the eyes inside it.
        roi_gray = gray[y:y+h ,x:x+w]
        roi_color = frame[y:y+h ,x:x+w]
        
        ##detectMultiScale method will find the place on the frame and it will return cordinates (x,y) , width , height.It can be multiple eyes on single frame also
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
        
        #now for loop will iterate throught every found eyes, and draw rectangle around it
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey) , (ex+ew ,ey+eh) ,(0,255,0) , 2)
    return frame

#now first take video frame from webcam it will capture all frame 
video_capture = cv2.VideoCapture(0)

#keep taking input from webcam untill we press key s
while True:
    
    #frame will read the frame of the webcam , and true will tell that frame is collecting imaage or not
    ret,frame = video_capture.read()
    
    #So untill we are getting image from webcam we will be getting the the output
    if ret==True:
        
        #detect function is taking two parameter one is coloured image and second gray image of that
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        #canvas will contain the detected image and then it will show it
        canvas = detect(frame,gray)
        cv2.imshow('Video',canvas)
        
    
    #press "s" button to close the window it come out of it
    if cv2.waitKey(1) & 0xFF==ord('s'):
        break
    
video_capture.release()
cv2.destroyAllWindows()






