#importing the libraries

import numpy as np 
import cv2 
from collections import deque 
from color_selector import color_detector
import time
import os



# calling the colour function 
color_detector()


# color_arr()

bpoints = [deque(maxlen = 512)] 
gpoints = [deque(maxlen = 512)] 
ypoints = [deque(maxlen = 512)] 
rpoints = [deque(maxlen = 512)] 

# Now to mark the pointers in the above colour array we introduce some index values Which would mark their positions  

blue_index = 0
green_index = 0
yellow_index = 0
red_index = 0

# The kernel is used for dilation of contour

kernel = np.ones((5, 5)) 

# The ink colours for the drawing purpose 
 
colors = [(255, 0, 0), (0, 255, 0), (0, 225, 255), (0, 0, 255)] 
colorIndex = 0

# Setting up the drawing board AKA The canvas 

paintWindow = np.zeros((471, 636, 3)) + 255

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE) 


# loading the installed/attached camera of the device 
 
cap = cv2.VideoCapture(0) 

while True: 

    # Reading the camera frame 
    ret, frame = cap.read() 
    # For saving
    # out = cv2.VideoWriter("Paint-Window.mp4", cv2.VideoWriter_fourcc(*'XVID'), 1, (frame.shape[1], frame.shape[0]))
    
    # Flipping the frame to see same side of the user  
    frame = cv2.flip(frame, 1) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # Getting the new positions of the trackbar and setting the new HSV values 

    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors") 
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors") 
    u_value = cv2.getTrackbarPos("Upper Value","Color detectors") 
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors") 
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors") 
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors") 
    Upper_hsv = np.array([u_hue, u_saturation, u_value]) 
    Lower_hsv = np.array([l_hue, l_saturation, l_value]) 

    # Adding the colour buttons to the live frame to choose color
    frame = cv2.rectangle(frame, (35, 1), (135, 65), (122, 122, 122), -1) 
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), -1) 
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), -1) 
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 255, 255), -1) 
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 255), -1) 

    cv2.putText(frame, "Clear All", (55, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA) 

    cv2.putText(frame, "Blue Color", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA) 
    
    cv2.putText(frame, "Green Color", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA) 

    cv2.putText(frame, "Yellow Color", (400, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 2, cv2.LINE_AA) 

    cv2.putText(frame, "Red Color", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA) 


    # masking out the pointer for it's identification in the frame 

    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv) 
    Mask = cv2.erode(Mask, kernel, iterations = 1) 
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel) 
    Mask = cv2.dilate(Mask, kernel, iterations = 1) 

    # Now contouring the pointers post identification 
    
    countours, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    centre = None

    # If there are any contours formed 
    if len(countours) > 0: 
        
        # sorting the contours for the biggest 
        countour = sorted(countours, key = cv2.contourArea, reverse = True)[0] 
        # Get the radius of the cirlce formed around the found contour   
        ((x, y), radius) = cv2.minEnclosingCircle(countour) 
        
        # Drawing the circle boundary around the contour 
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
        
        # Calculating the centre of the detected contour 
        M = cv2.moments(countour) 
        centre = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) 
        
        # Now checking if the user clicked on another button on the screen (the 4 buttons that were mentioned Y,G,B,R and clear all)
        if centre[1] <= 65: 
            
            # Clear Button 
            if 35 <= centre[0] <= 135: 
                bpoints = [deque(maxlen = 512)] 
                gpoints = [deque(maxlen = 512)] 
                ypoints = [deque(maxlen = 512)] 
                rpoints = [deque(maxlen = 512)] 

                blue_index = 0
                green_index = 0
                yellow_index = 0
                red_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= centre[0] and centre[0] <= 255: 
                colorIndex = 0 # Blue 
                    
            elif 275 <= centre[0] and centre[0] <= 370: 
                colorIndex = 1 # Green 
            elif 390 <= centre[0] and centre[0] <= 485: 
                colorIndex = 2 # Yellow
            elif 505 <= centre[0] and centre[0] <= 600: 
                colorIndex = 3 # Red 
        else : 
            if colorIndex == 0: 
                bpoints[blue_index].appendleft(centre) 
            elif colorIndex == 1: 
                gpoints[green_index].appendleft(centre) 
            elif colorIndex == 2: 
                ypoints[yellow_index].appendleft(centre) 
            elif colorIndex == 3: 
                rpoints[red_index].appendleft(centre) 
                
    # Appending the next deques if nothing is detected

    else: 
        bpoints.append(deque(maxlen = 512)) 
        blue_index += 1
        gpoints.append(deque(maxlen = 512)) 
        green_index += 1
        ypoints.append(deque(maxlen = 512)) 
        yellow_index += 1
        rpoints.append(deque(maxlen = 512)) 
        red_index += 1

    # Drawing the lines of every colour on the canvas and the track frame window
    
    points = [bpoints, gpoints, ypoints, rpoints] 
    for i in range(len(points)): 
        for j in range(len(points[i])): 
            for k in range(1, len(points[i][j])): 
                if points[i][j][k - 1] is None or points[i][j][k] is None: 
                    continue
                    
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 25) 
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 25) 

    key = cv2.waitKey(1)            
    if key & 0xFF == ord('f'):            
        cv2.imwrite("last_frame.jpg", paintWindow)
        




    # Displaying/running all the 3 windows 
    cv2.imshow("Live Tracking", frame) 
    cv2.imshow("Paint", paintWindow) 
    cv2.imshow("mask", Mask) 
    
    # For quitting/breaking the loop - press and hold ctrl+q twice 
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

# Releasing the camera and all the other resources of the device  
cap.release() 
cv2.destroyAllWindows() 
