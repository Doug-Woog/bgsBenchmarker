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
import sys
#sys.path.insert(0, "/media/DATA/gym_ai/GymAI/src")
from BoundingBoxUtilities import BoundingBoxUtilities  as bbox_utils
from VisualizationTools import VisualizationTools as visualize 
root="C:\...\PETS Data"
cap = cv2.VideoCapture(os.path.join(root, "S2_L1_Time_12-34_View_001_2fps.mp4")) 
import xml.etree.ElementTree as ET

ORIGINAL_SHAPE = (576, 768) # of PETS videos

frame_boxes={}

def PETS_xml_to_dictionary(filepath):
    """
    Data Schema:
    frame_boxes = { framenumber1 : { id1 : (x,y,w,h) , id2 : (x,y,w,h),  ...} 
                    , ... } 
                   }
    """
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Show all unique elements in the tree: set([elem.tag for elem in root.iter()])
    
    for frame in root:
        #print(frame.tag, frame.attrib)
        frame_number = frame.attrib["number"]
        this_frames_data={} 
        for objlist in  frame.getchildren():
            for obj in objlist.getchildren():
                box_id=obj.attrib["id"]
                # every id is assumed to only have one box in a frame, by this zero index.
                box_data = obj.getchildren()[0].attrib 
                x_mid=float(box_data["xc"])
                y_mid=int(float(box_data["yc"]))
                w=int(float(box_data["w"]))
                h=int(float(box_data["h"]))
                x=int(x_mid-w/2.)
                y=int(y_mid-h/2.)
                this_frames_data[box_id] = (x,y,w,h)
    
        frame_boxes[str(frame_number)] = this_frames_data
   
    return frame_boxes 

frame_boxes = PETS_xml_to_dictionary('PETS2009-S2L1-cropped.xml')

last_frame= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
box_frame_num = 0
while True:
    ret, img = cap.read()
    if ret and (img is not None):

        
        current_frame_data=frame_boxes[str(box_frame_num)]
        with open("frame"+str(current_frame)+".txt",'w') as f:
            for box_id in current_frame_data.keys():
                box = current_frame_data[box_id]
                box = bbox_utils.scaleBoundingBoxes([box,], ORIGINAL_SHAPE,img.shape)[0]
                x,y,w,h = box
            
                f.write("Person "+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 100), 2)
                img=visualize.addOverlayText(f"{box_id}", img, x, y)
            
        cv2.imshow("window", img)
            
    if cv2.waitKey(100) & 0xFF == ord('q'):  # press q to quit
        break
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break
    if current_frame == last_frame:
        break
    current_frame+=1
    box_frame_num += 2
    if (current_frame % 100) == 0:
        print("Current Frame:", current_frame)
    
cap.release()
cv2.destroyAllWindows()
