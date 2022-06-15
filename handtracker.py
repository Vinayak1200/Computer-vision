import cv2           #Library to capture images and video data 
import mediapipe as mp  #Library for hand detection from the video data collected by opencv
import time

cap = cv2.VideoCapture(0)       #cap stores the webcam video as a video object 

mpHands = mp.solutions.hands
hands = mpHands.Hands()         #Function for hand detection and tracking
mpDraw = mp.solutions.drawing_utils    #This is for drawing the 21 points in the final image and display it


while True:
    success, img = cap.read()    #read() returns a tuple, the first one indicating the return value, 
                                 #and the second one for storing the frame in the form of image

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #This converts img into rgb format, since hands function takes RGB image as input
    results = hands.process(imgRGB)
    lmlist = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks((img), handLms, mpHands.HAND_CONNECTIONS)      #This draws the 21 points on every hand detected 
    cv2.imshow('Image', img)                                                     #in the image and displays those points

    cv2.waitKey(1)            #This function takes the number of milliseconds an image must be displayed 
                              #Here, the input is 1 millisecond

    