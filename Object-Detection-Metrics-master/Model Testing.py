import cv2
import os
import numpy as np
from imutils.video import FPS
import sys
import csv
from PeopleDetectionMetrics import calculate_metrics
bgs_dir = "C:\\Users\Daniel\\bgslibrary\\bgslibrary"
data_root =  os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.insert(0,data_root)
sys.path.insert(0, bgs_dir)
from MaskPostProcessing import MaskPostProcessing
from BoundingBoxUtilities import BoundingBoxUtilities as bbox_utils
from VisualizationTools import VisualizationTools as visualize
import bgs
INPUT_FRAME_SHAPE = (270, 360, 3)
INPUT_FRAME_W = 360
INPUT_FRAME_H = 270
detection_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections')

def pretrain_on_background_vid(fgbg,PRETRAINING_NUMBER_OF_FRAMES,display=True,mog=False):
    initial_background_vid = cv2.VideoCapture(os.path.join(data_root,'S0_BG_View_001_Time_13-32-bestmatch.mp4'))

    # Pretrain on 100 frames
    frame_num = 0
    while(True):
        # Capture frame-by-frame
        ret, img = initial_background_vid.read()
        
       
        # This is the safe way to exit loop if end of video/problem with frame
        if (not ret) or (img is None):
            break
        
        if img.shape != INPUT_FRAME_SHAPE:
            img = cv2.resize(img, (INPUT_FRAME_W, INPUT_FRAME_H))
            assert img.shape == INPUT_FRAME_SHAPE
        
        if mog == True:
            fgbg.apply(img, learningRate = 0.001)
        else:
            fgbg.apply(img)
        
        if display == True:
            cv2.imshow('img', img)
        
            k = cv2.waitKey(1)
        
            if k == ord('q'):
                break
        
        frame_num += 1
        
        if frame_num == PRETRAINING_NUMBER_OF_FRAMES:
            break
        
    initial_background_vid.release()
    cv2.destroyAllWindows()    
    return fgbg

def train_model(model,write=False,display=True,mog=False):
    foreground_vid = cv2.VideoCapture(os.path.join(data_root, 'S2_L1_Time_12-34_View_001_2fps.mp4'))
    frame_num = 0
    fps_manager = FPS().start() # start timing FPS

    while(True):
        # Capture frame-by-frame
        ret, img = foreground_vid.read()
        
        # This is the safe way to exit loop if end of video/problem with frame
        if (not ret) or (img is None):
            break
    
        if img.shape != INPUT_FRAME_SHAPE:
            img = cv2.resize(img, (INPUT_FRAME_W, INPUT_FRAME_H))
            assert img.shape == INPUT_FRAME_SHAPE
        

        if mog == True:
            fgmask = model.apply(img, learningRate = 0.0005)
            #fgmask2 = model2.apply(img)
        else:
            fgmask = model.apply(img)
        
        fgmask_enhanced = MaskPostProcessing.apply_filter('filter3_low_recall_low_noise_entrance', fgmask)
        fgmask_enhanced, boxes = MaskPostProcessing.findContours(fgmask_enhanced, lower_area=0, upper_area=np.inf)
        """
        fgmask_enhanced2 = MaskPostProcessing.apply_filter('filter3_low_recall_low_noise_entrance', fgmask2)
        fgmask_enhanced2, boxes2 = MaskPostProcessing.findContours(fgmask_enhanced2, lower_area=0, upper_area=np.inf)
        """
        if write == True:
            with open(detection_folder+"\frame"+str(frame_num)+".txt",'w') as f:
                for box in boxes:
                    f.write("Person "+'1 '+str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+'\n')

        if display == True:
            #img2 = copy.deepcopy(img)
            cv2.imshow('Input Image', img)
            cv2.imshow('Output Mask', fgmask)
            cv2.imshow('Enhanced Mask', fgmask_enhanced)
            if mog==False:
                cv2.imshow('Background',model.getBackgroundModel())
            
            visualize.drawBoxes(img, boxes)
            cv2.imshow('Boxed Mask', img)
            """
            cv2.imshow('Output Mask Mixture', fgmask2)
            cv2.imshow('Enhanced Mask Mixture', fgmask_enhanced2)
            #cv2.imshow('Background Mixture',model2.getBackgroundModel())
            
            visualize.drawBoxes(img2, boxes2)
            cv2.imshow('Boxed Mask Mixture', img2)
            """
            k = cv2.waitKey(1)
    
            if k == ord('q'):
                #cv2.waitKey(1)
                break
    
        fps_manager.update()
        frame_num+=1
 
    foreground_vid.release()
    cv2.destroyAllWindows()

    # Measure performance
    fps_manager.stop()
    time_elapsed = fps_manager.elapsed()
    print("[STDOUT] elapsed time: {:.2f}".format(time_elapsed))
    print("[STDOUT] approx. FPS: {:.2f}".format(fps_manager.fps()))
    return fps_manager.fps()

#import copy

def test(models,IOU,WithMOG=True,AP=False):
    
    with open(models,'r') as a:
        results = dict()
        results["Headers"] = []
        results["Headers"].append("FPS")
        for thresh in IOU:
            if AP == True:
                results["Headers"].append("AP (IOU = "+str(thresh)+")")
            results["Headers"].append("Precision (IOU = "+str(thresh)+")")
            results["Headers"].append("Recall (IOU = "+str(thresh)+")")
            
        for model in a.readline()[1:-1].split("', '"): #Removed T2FMRF_UM model
            #if model[:2] != "LB" and model[:5] != "Multi" and model != "VuMeter":
            results[model] = []
            model_test = getattr(bgs, model)() 
            model_test = pretrain_on_background_vid(model_test,100,display=False)
            results[model].append(train_model(model_test,write=True,display=True))
            for threshold in IOU:
                if AP == True:
                    for result in calculate_metrics(IOUThresh=threshold):
                        results[model].append(result)
                else:
                    for result in calculate_metrics(IOUThresh=threshold)[-2:]:
                        results[model].append(result)
                #print(results[model])
    if WithMOG==True:
        mog_model = cv2.createBackgroundSubtractorMOG2( history=200,
                                           varThreshold=200,
                                        detectShadows=True)
        results["BackgroundSubtractorMOG2"] = []
        model_test = pretrain_on_background_vid(mog_model,100,display=False,mog=True)
        results["BackgroundSubtractorMOG2"].append(train_model(mog_model,write=True,display=False,mog=True))
        for threshold in IOU:
            for result in calculate_metrics(threshold):
                results["BackgroundSubtractorMOG2"].append(result)

    with open("Test_Results IOU Threshold ="+str(IOU)+"AP="+str(AP)+".csv", "w",newline='') as outfile:
       writer = csv.writer(outfile)
       writer.writerow(results.keys())
       writer.writerows(zip(*results.values()))

test("TEst Models.txt",[0.25,0.5,0.75],AP=False,WithMOG=False) #Available bgs models.txt

"""
mog_model = cv2.createBackgroundSubtractorMOG2( history=200,
                                                   varThreshold=200,
                                                detectShadows=True)
        mixture_model = bgs.MixtureOfGaussianV2()
        pretrain_on_background_vid(mog_model,100,display=False,mog=True)
        pretrain_on_background_vid(mixture_model,100,display=False)
        train_model(mog_model,mixture_model,write=False,display=True,mog=True)
"""