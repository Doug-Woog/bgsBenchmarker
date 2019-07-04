
import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
import numpy
import os
def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath, 'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    groundtruthboxes = dict()
    detectionboxes = dict()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for frame_num, f in enumerate(files):
        if frame_num < 20:
            continue
        nameOfImage = f.replace(".txt", "")
        groundtruthboxes[nameOfImage] = BoundingBoxes()
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (360, 270),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
            groundtruthboxes[nameOfImage].addBoundingBox(bb)
            """
            bb2 = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (360, 270),
                BBType.Detected,
                1,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb2)
            """
            #detectionboxes[nameOfImage].addBoundingBox(bb2)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, 'detections')
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    
    for frame_num, f in enumerate(files):
        if frame_num < 20:
            continue
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        detectionboxes[nameOfImage] = BoundingBoxes()
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (360, 270),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
            detectionboxes[nameOfImage].addBoundingBox(bb)
        fh1.close()
    
    os.chdir('..')
    return allBoundingBoxes,groundtruthboxes,detectionboxes


def createImages(dictGroundTruth, dictDetected): #This function is not maintained and so will likely not work.
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 360
    height = 270
    # Loop through the dictionary with ground truth detections
    foreground_vid = cv2.VideoCapture(os.path.join(os.path.join( os.path.dirname( __file__ ), '..' ), 'S2_L1_Time_12-34_View_001_2fps.mp4'))
    for key in range(1,397):

        ret, img = foreground_vid.read()
        # This is the safe way to exit loop if end of video/problem with frame
        if (not ret) or (img is None):
            break
        if img.shape != (270, 360, 3):
            img = cv2.resize(img, (INPUT_FRAME_W, INPUT_FRAME_H))
            assert img.shape == (270, 360, 3)
        gt_boundingboxes = dictGroundTruth[("frame"+str(key))]
        img = gt_boundingboxes.drawAllBoundingBoxes(img,("frame"+str(key)))
        detection_boundingboxes = dictDetected[("frame"+str(key))]
        img = detection_boundingboxes.drawAllBoundingBoxes(img,("frame"+str(key)))
        # Show detection and its GT
        cv2.imshow("All boxes", img)
        cv2.waitKey()

# Uncomment the line below to generate images based on the bounding boxes
#boundingboxes,dictGroundTruth,dictDetected = getBoundingBoxes()
#createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics

def calculate_metrics(IOUThresh = 0.5):
    evaluator = Evaluator()

    metricsPerClass = evaluator.GetPascalVOCMetrics(
        getBoundingBoxes()[0],  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=IOUThresh,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    #print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics

    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        print("Precision: "+str(precision[-1])) #True Precision Value
        print("Recall: "+str(recall[-1])) #True Recall Value
        return  average_precision, precision[-1], recall[-1],
        #print('%s: %f' % (c, average_precision))
        #print("\nRecall Average ="+str(numpy.mean(mc['recall'])))

