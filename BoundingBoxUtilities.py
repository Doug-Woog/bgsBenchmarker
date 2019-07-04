'''
\package docstring

 Set of functions for manipulating bounding boxes.

 Created on December 4th 2018 by Benjamin Mark Lowe
'''

import textwrap
import os
import numpy as np
import itertools


class BoundingBoxUtilities(object):
    '''
    \brief Reads in bounding boxes from a file in relative coordinates (YOLO format) and scales to the appropriate frame size.
    \param[in] txt_path A YOLO format file including bounding boxes.
    \param[in] frame_shape The shape of the desired image, boxes will be scaled to this dimension.
    \returns boxes A list of the bounding boxes scaled to img_shape dimensions.
    '''
    @staticmethod
    def getBoxesFromFile(txt_path, img_shape, bb_format="yolo"):
        height, width = img_shape[0], img_shape[1]

        boxes=[]

        if os.path.isfile(txt_path):
            try:
                with open(txt_path) as f:
                    content = f.readlines()
            except IOError:
                raise IOError(f"Could not read file: {txt_path}")
            for line in content:
                values_str = line.split()
                if bb_format == 'yolo':
                    class_index, x_center, y_center, x_width, y_height = map(float, values_str)
                    x = x_center - (x_width/2) #relative
                    y = y_center - (y_height/2)
                    x1=int(x*width)
                    y1=int(y*height)
                    w=int(x_width*width)
                    h=int(y_height*height)

                    if x_center == int(x_center):
                        error = ("You selected the 'yolo' format but your labels "
                                "seem to be in a different format. Consider "
                                "removing your old label files.")
                        raise Exception(textwrap.fill(error, 70))
                '''
                #NOT YET IMPLEMENTED
                elif bb_format == 'voc':
                    try:
                        x1, y1, x2, y2, class_index = map(int, values_str)
                    except ValueError:
                        error = ("You selected the 'voc' format but your labels "
                                "seem to be in a different format. Consider "
                                "removing your old label files.")
                        raise Exception(textwrap.fill(error, 70))
                    x1, y1, x2, y2 = x1-1, y1-1, x2-1, y2-1

                elif bb_format == 'opencv':
                    try:
                        class_index, x, y, w, h = map(int, values_str)
                    except ValueError:
                        error = ("You selected the 'opencv' format but your labels "
                                "seem to be in a different format. Consider "
                                "removing your old label files.")
                        raise Exception(textwrap.fill(error, 70))
                    x1, y1, x2, y2 = x, y, x+w, y+h
                '''

                boxes.append((x1,y1, w, h))

        return boxes


    '''
    \brief Scales input bounding boxes from the scale of original_frame to the scale of shrunk_frame.
    \param[in] boxes Boxes a list of bounding boxes in (x,y,w,h) format
    \param[in] original_shape Shape of the image used to get the bounding boxes.
    \param[in] shrunk_frame_shape Shape of the resulting shrunked image, for which we desire bounding boxes to be scaled to.
    \returns Scaled bounding boxes.
    '''
    @staticmethod
    def scaleBoundingBoxes(boxes, original_shape, shrunk_frame_shape):
        # Obtain boxes at the scale of the shrunk frame so only motion
        # in these regions is used
        shrunk_boxes=[]
        shrunk_frame_w=shrunk_frame_shape[0]
        shrunk_frame_h=shrunk_frame_shape[1]
        original_w=original_shape[0]
        original_h=original_shape[1]

        scale_w=shrunk_frame_w/original_w
        scale_h=shrunk_frame_h/original_h
        for (x, y, w, h) in boxes:
            new_width = int(w*scale_w)
            new_height = int(h*scale_h)

            # A box with no width was identified, if this is the case, may want to consider a larger area requirement for boxes. These boxes are removed.
            if new_width > 0 and new_height > 0:
                shrunk_boxes.append((int(x*scale_w), int(y*scale_h), new_width, new_height))

        return shrunk_boxes


    '''
    \brief Given an image and a bounding box, crop the image to the bounding box with a border of `buffer` up to the edge of the image.
    \param[in] img Numpy image array
    \param[in] box Boxes a list of bounding boxes in (x,y,w,h) format
    \param[in] buffer Dimensions (width=height) of the border to add around the box in units of pixels.
    \returns Cropped images of the bounding box.
    '''
    @staticmethod
    def cropWithBuffer(img, box, buffer):

        (x, y, w, h) = box

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        y_len,x_len = img.shape[0], img.shape[1]

        #Crop the image with a margin of enlarge_box
        y_pos_lower=y-buffer
        y_pos_upper=y+h+buffer
        x_pos_lower=x-buffer
        x_pos_upper=x+w+buffer

        #Ensure it does not crop beyond boundaries of image
        if y-buffer < 0:
            y_pos_lower = 0
        if y+h+buffer >= y_len:
            y_pos_upper = y_len
        if x-buffer < 0:
            x_pos_lower = 0
        if x+w+buffer >= x_len:
            x_pos_upper = x_len

        cropped = img[y_pos_lower:y_pos_upper,x_pos_lower:x_pos_upper]

        # The xy corner of the box is frame is no longer the origin of the
        # coordinate system using for the bounding boxes
        # So return the offset for later use.
        buffer_xy_offset  = int(x - x_pos_lower), int(y - y_pos_lower)


        return cropped, buffer_xy_offset

    '''
    \brief Given a bounding box, expand it by a buffer without going off the frame.
    \param[in] img Numpy image array
    \param[in] input_box A bounding box in (x,y,w,h) format
    \param[in] buffer Dimensions (width=height) of the border to add around the box in units of pixels.
    \returns box An expanded bounding box in (x,y,w,h) format
    '''
    @staticmethod
    def expandBoxByBuffer(img, input_box, buffer):

        (x, y, w, h) = input_box

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        #Crop the image with a margin of enlarge_box
        y_pos_lower=y-buffer
        x_pos_lower=x-buffer

        #Ensure it does not crop beyond boundaries of image
        if y-buffer < 0:
            y_pos_lower = 0
        if x-buffer < 0:
            x_pos_lower = 0

        box = (x_pos_lower, y_pos_lower, w+buffer*2, h+buffer*2)
        return box


    """
    \brief Takes a list of bounding boxes in (x,y,w,h) format and converts them to a list of centroids in (x_center,y_center) format.
    \param[in] boxes list of bounding boxes in (x,y,w,h) format
    \returns centroids list of centroids in (x_center,y_center)
    """
    @staticmethod
    def boxesToCentroids(boxes):
        centroids = []
        for (x,y,w,h) in boxes:
            centroids.append((int(x+w//2), int(y+h//2)) )
        return centroids

    """
    \brief Takes a box in (x,y,w,h) format and converts it to a centroid in (x_center,y_center) format.
    \param[in] boxes bounding box in (x,y,w,h) format
    \returns centroid centroid in (x_center,y_center)
    """
    @staticmethod
    def boxToCentroids(box):
        (x,y,w,h) = box
        return (int(x+w//2), int(y+h//2))


    """"
    \brief Given a list of boxes in (x,y,w,h) format, returns a list of the same boxes but with fixed width and height and centered on the original box centroid.
    \param[in] boxes List of bounding boxes (x,y,w,h) format
    \param[in] fixed_w Number of pixels to set each box width to.
    \param[in] fixed_h Number of pixels to set each box width to.
    \returns boxes List of bounding boxes (x,y,w,h) format
    """
    @staticmethod
    def makeFixedSizeBoxFromBoxes(boxes, fixed_w=25, fixed_h=50):
        centroids = BoundingBoxUtilities.boxesToCentroids(boxes)
        fixed_size_boxes = []
        for (x,y) in centroids:
            fixed_size_boxes.append((x-fixed_w//2,y-fixed_h//2,fixed_w,fixed_h))
        return fixed_size_boxes

    """
    \brief Boxes with a width larger than width_threshold pixels will be split in half. If recursive = True, then this will be done recursively. Implementation is performed by removing the large box and appending the new boxes to the end of the list with width `originalWidth/2`.
    \param[in] boxes List of bounding boxes (x,y,w,h) format
    \param[in] width_threshold Threshold number of pixels, if a box is greater than this it will be split in half.
    \param[in] recursive Bool. Whether to keep splitting recursively until all boxes are below width_threshold.
    \returns boxes List of bounding boxes (x,y,w,h) format
    \note This function is similar to splitLargeBoxes() but differ..This function splits the box as many times as needed until below the threshold, whereas splitLargeBoxes() takes the original width and just splits it equally to as many boxes as needed.  So if more than one split occurs, the result will likely differ between them.
    """
    @staticmethod
    def splitLargeBoxesInHalf(boxes, width_threshold=25, recursive=False):
        removal_indexes = []
        new_boxes = []
        for i, (x,y,w,h) in enumerate(boxes):
            if w > width_threshold:
                removal_indexes.append(i)
                box1 = (x,y,w//2,h)
                box2 = (x+w//2, y, w//2, h)
                if recursive:
                    boxes.append(box1)
                    boxes.append(box2)
                else:
                    new_boxes.append(box1)
                    new_boxes.append(box2)
        boxes = np.delete(np.array(boxes), removal_indexes, axis=0).tolist()

        if not recursive:
            boxes += new_boxes

        return boxes


    """
    \brief Boxes with a width larger than width_threshold pixels will be split into `n` side by side boxes where  `n` is the number of whole boxes that would fit within the box at that width_threshold. This is done by removing the large box and appending `n` new boxes to the end of the list with width `width_threshold/n`. `
    \param[in] boxes List of bounding boxes (x,y,w,h) format
    \param[in] width_threshold Threshold number of pixels, if `n` boxes of width=width_threshold can fit into the box's width, then divide it equally into `n` boxes with width equal to the original box divided by `n`.
    \param[in] max_splits Optional integer argument - if provided then it will set the maximum number of times the box is split into sub-boxes. e.g.max_splits=1 will split any box larger than `width_threshold` in half.
    \returns boxes List of bounding boxes (x,y,w,h) format
    \note This function is similar to splitLargeBoxesInHalf() but differs. This function takes the original width and splits it equally to as many boxes as needed. By contrast,  splitLargeBoxesInHalf() splits it in half many times until below the threshold. So if more than one split occurs, the result will likely differ between them.
    """
    @staticmethod
    def splitLargeBoxes(boxes, width_threshold=25, max_splits=None):
        removal_indexes = [int(i) for i in np.argwhere((np.array(boxes)[:,2] // width_threshold) >= 1)]
        for idx in removal_indexes:
            (x,y,w,h) = boxes[idx]
            boxes_required = w // width_threshold # number of boxes that fit inside of the box.
            if boxes_required >= 1:
                if max_splits:
                    boxes_required = max_splits + 1
                for i in range(boxes_required):
                    newbox = (x+i*(w//boxes_required),y,w//boxes_required,h)

                    boxes.append(newbox)
        boxes = np.delete(np.array(boxes), removal_indexes, axis=0).tolist()
        return boxes

    '''
    \brief Test if the point is inside the box
    \param[in] box Bounding box to be tested (x,y,w,h)
    \param[in] point Point to be tested (x,y)
    \returns boolean True if the point is inside the box and false otherwise
    \raises AssertionError in case the box or point variables are not in the right format
    '''
    @staticmethod
    def isInside(box, point):
        if(len(box)!=4):
            raise AssertionError("Invalid box format. Please use (x,y,width,height)")
        if(len(point)!=2):
            raise AssertionError("Invalid point format. Please use (x,y)")
        if(point[0]>=box[0] and
           point[0]<box[0]+box[2] and
           point[1]>=box[1] and
           point[1]<box[1]+box[3]):

            return True
        return False

    '''
    \brief Given 2 points, one in the past and one in the present, returns if the object entered or exited the box and from which side. In case the object have not entered nor exited the box, None is returned
    \param[in] box Bounding box to be tested (x,y,w,h)
    \param[in] pt_past Point in the past (x,y)
    \param[in] pt_present Point in the present (x,y)
    \returns array Array with the answers (string, string) with the first value being if the object entered or exited and the second value being to which side it exited. If the object haven't entered nor exited the box, None is returned.
    \note The possible strings for the first value are "entered" or "exited". The values for the second string are "north", "east", "west" and "south". If the object moved in diagonal, north and south have priority.
    '''
    @staticmethod
    def enterExitedDirection(box, pt_past, pt_present):
        #test if the point entered or exited
        past_inside = BoundingBoxUtilities.isInside(box, pt_past)
        present_inside = BoundingBoxUtilities.isInside(box, pt_present)
        #if haven't entered nor exited both values will be the same
        if(past_inside == present_inside):
            return None
        if((not past_inside) and (present_inside)):
            entered = "entered"
        else:
            entered = "exited"

        #now lets find the direction
        if(not past_inside):
            pt_outside = pt_past
        else:
            pt_outside = pt_present

        if(pt_outside[1]<box[1]):
            direction = "north"
        elif(pt_outside[1]>box[1]+box[3]):
            direction = "south"
        elif(pt_outside[0]<box[0]):
            direction = "west"
        else:
            direction = "east"

        return (entered, direction)

    '''
    \brief Returns intersection-over-union of two bounding boxes.
    \param[in] A Box A. Format: (x, y, w, h)
    \param[in] B Box B. Format: (x, y, w, h)
    \returns The intersection over union of the two bounding boxes
    '''
    @staticmethod
    def bbIntersectionOverUnion(A, B):
        # Internally use style of box representation
        x, y, w, h = A
        boxA = x, y, x + w, y + h
        x, y, w, h = B
        boxB = x, y, x + w, y + h

    	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

    	# compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

    	# compute the area of both the prediction and ground-truth
    	# rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    	# compute the intersection over union by taking the intersection
    	# area and dividing it by the sum of prediction + ground-truth
    	# areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

    	# return the intersection over union value
        return iou


    '''
    \brief Convert box(es) format from (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max)
    \param[in] box Bounding box(es) as numpy array with shape (n_boxes, 4)
    \returns A new box in   (x_min, y_min, x_max, y_max)  format, as a numpy array of shape (n_boxes, 4)
    '''
    @staticmethod
    def xywh2xyxy(box):
        newbox = np.empty_like(box)
        newbox[:,0] = box[:,0]
        newbox[:,1] = box[:,1]
        newbox[:,2] = box[:,0] + box[:,2]
        newbox[:,3] = box[:,1] + box[:,3]
        return newbox

    '''
    \brief  Convert box(es) format from (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)
    \param[in] box Bounding box(es) as numpy array with shape (n_boxes, 4)
    \returns A new box in  (x_min, y_min, width, height)  format, as a numpy array of shape (n_boxes, 4)
    '''
    @staticmethod
    def xyxy2xywh(box):
        newbox = np.empty_like(box)
        newbox[:, 0] = box[:, 0]
        newbox[:, 1] = box[:, 1]
        newbox[:, 2] = box[:, 2] - box[:, 0]
        newbox[:, 3] = box[:, 3] - box[:, 1]
        return newbox

    '''
    \brief Computes the intersection area of each pair of boxes along the n_boxes dimension of the boxes.
    \param[in] boxA Bounding box(es) as numpy array with shape (n_boxes, 4). x,y,w,h format.
    \param[in] boxB Bounding box(es) as numpy array with shape (n_boxes, 4). x,y,w,h format.
    \returns interArea The intersection area of each pair of boxes along the n_boxes dimension.
    \note boxA and boxB should be same dimensions.
    '''
    @staticmethod
    def comp_intersection(boxA, boxB):
        boxA = BoundingBoxUtilities.xywh2xyxy(boxA)
        boxB = BoundingBoxUtilities.xywh2xyxy(boxB)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[:,0], boxB[:,0])
        yA = np.maximum(boxA[:,1], boxB[:,1])
        xB = np.minimum(boxA[:,2], boxB[:,2])
        yB = np.minimum(boxA[:,3], boxB[:,3])
        # compute the area of intersection rectangle
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        return interArea

    '''
    \brief Computes area of box(es)
    \param[in] box Bounding box(es) as numpy array with shape (n_boxes, 4). x,y,w,h format.
    \returns Area of box(es)
    '''
    @staticmethod
    def comp_area(box):
        return box[:,2] * box[:,3]

    '''
    \brief Computes intersection over union of box(es)
    \param[in] boxA Bounding box(es) as numpy array with shape (n_boxes, 4). x,y,w,h format.
    \param[in] boxA Bounding box(es) as numpy array with shape (n_boxes, 4). x,y,w,h format.
    \returns iou intersection over union of box(es)
    \note boxA and boxB should be same dimensions.
    '''
    @staticmethod
    def comp_iou(boxA, boxB):
        interArea = BoundingBoxUtilities.comp_intersection(boxA, boxB)
        boxAArea = BoundingBoxUtilities.comp_area(boxA)
        boxBArea = BoundingBoxUtilities.comp_area(boxB)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    '''
    \brief Compute iou values between all pairs of boxes from boxes_A and boxes_B
    \param[in] boxes_A numpy array of shape (n_boxes,4), box format is (x_min, y_min, width, height)
    \param[in] boxes_B numpy array of shape (n_boxes,4), box format is (x_min, y_min, width, height)
    \returns iou_matrix iou matrix of shape (len(boxes_A), len(boxes_B)). Element (i, j) is IOU of boxes_A[i, :] and boxes_B[j, :]
    '''
    @staticmethod
    def comp_iou_matrix(boxes_A, boxes_B):
        n_A = len(boxes_A)
        n_B = len(boxes_B)
        idx_A = np.arange(n_A)
        idx_B = np.arange(n_B)
        pairs = itertools.product(idx_B, idx_A)
        idx_pair_1, idx_pair_2 = list(zip(*pairs))
        # First and second elements of each pair
        first = boxes_B[idx_pair_1, :]
        second = boxes_A[idx_pair_2, :]
        # iou matrix
        iou_values = BoundingBoxUtilities.comp_iou(first, second)
        iou_matrix = iou_values.reshape(n_B, n_A).transpose()
        return iou_matrix


    '''
    \brief Converts a box from YOLO format: (x_center,y_center,x_width,y_height) normalised between 0 and 1, to absolute pixels scale, based on the image shape `img_shape` provided.
    \param[in] yolo_box A bounding box (x_center,y_center,x_width,y_height) normalised between 0 and 1
    \param[in] img_shape The shape of the numpy array image the boxes are to be projected onto.
    \returns opencv_box (x,y,w,h) format bounding box with units of absolute pixels.
    '''
    @staticmethod
    def yolo_to_opencv_bbox_format(yolo_box, img_shape):
        height, width = img_shape[0], img_shape[1]
        x_center, y_center, x_width, y_height = yolo_box
        x = x_center - (x_width/2) #relative
        y = y_center - (y_height/2)
        x1=int(x*width)
        y1=int(y*height)
        w=int(x_width*width)
        h=int(y_height*height)
        opencv_box = (x1,y1,w,h)
        return opencv_box
