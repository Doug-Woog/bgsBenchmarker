#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:34:10 2018

@author: ben
"""
import cv2 
import os
import imutils 
import glob

base="."

## Crowd Example
#root=os.path.join(base, "S1/L1/Time_13-59/View_001") 
#f= 'frame_%04d.jpg' 
#out_name= f'S1_L1_Time_13-59_View_001.mp4'

## Background Images
#root=os.path.join(base, "S0_BG/Crowd_PETS09/S0/Background/View_001/Time_13-19/")
#f='%08d.jpg'
#out_name= f'S0_BG_View_001_Time_13-59.mp4'

#root=os.path.join(base, "S0_BG/Crowd_PETS09/S0/Background/View_001/Time_13-32/")
#f='%08d.jpg'
#out_name= f'S0_BG_View_001_Time_13-32.mp4'


#root=os.path.join(base, "S0_BG/Crowd_PETS09/S0/Background/View_001/Time_13-38/")
#f='%08d.jpg'
#out_name= f'S0_BG_View_001_Time_13-38.mp4'

## 
root=os.path.join(base, "View_001/")
f= 'frame_%04d.jpg' 
out_name= f'S2_L1_Time_12-34_View_001_2fps.mp4'

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
target_h=270
target_w=360

# input FPS is 5.0 I think (based on http://www.cvg.reading.ac.uk/PETS2009/a.html#s2)
# output FPS is approximately 2.0
out = cv2.VideoWriter(out_name, fourcc, 2.0, (target_w, target_h))

cap = cv2.VideoCapture(os.path.join(root,f))

last_frame=len(glob.glob(root+"*"))
current_frame=0
while True:
    ret, img = cap.read()
    if ret and (img is not None):
        # Input FPS is 5, so skip every 3 frames to get roughly 2 FPS
        if current_frame % 2 == 0:
            img = cv2.resize(img, (360, 270))
            img=imutils.resize(img, width=360)
            #cropped=img[0:target_h, 0:target_w]

            #print(img.shape)
            #cv2.imshow("window", cropped)
            out.write(img)
            cv2.imshow("window", img)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break
    if current_frame == last_frame:
        break
    
    current_frame+=1
    
    if (current_frame % 100) == 0:
        print("Current Frame:", current_frame)
    
out.release()
cap.release()
cv2.destroyAllWindows()
