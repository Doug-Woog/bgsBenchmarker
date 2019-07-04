import cv2
import numpy as np
import sys
sys.path.insert(0, '../../')

'''
\brief Implementation of common mask post-processing algorithms like noise removal, dilate and erode.
\note Since there is no need to save several copies of masks post processings, this class is an abstract one
'''
class MaskPostProcessing(object):

    '''
    \brief Applies morphological dilation to the image with an circular kernel
    \note The image need to be a 1 channel image
    \raises AssertionError in case the provided image is not single channel or is not binary
    \param[in] img Numpy image to be processed
    \param[in] kernel_size The size of the kernel used. if set to x the kernel size will be (x, x)
    \param[in] iterations Number of iterations of the convolution
    \returns Processed image
    '''
    @staticmethod
    def dilateImage(img, kernel_size=10, iterations=2):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accepted')
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(img, dilation_kernel, iterations=iterations)

    '''
    \brief Applies morphological dilation to the image with an vertically stretched elipse kernel. 
    \note The image need to be a 1 channel image
    \raises AssertionError in case the provided image is not single channel or is not binary
    \param[in] img Numpy image to be processed
    \param[in] kernel_size The size of the kernel used. if set to x the kernel size will be (x, x)
    \param[in] iterations Number of iterations of the convolution
    \returns Processed image
    '''
    @staticmethod
    def dilateImageVertically(img, kernel_size=6, iterations=1, clip_edge_columns=1):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accepted')
        dilation_kernel = customVerticalDilationKernel(kernel_size, clip_edge_columns)
        return cv2.dilate(img, dilation_kernel, iterations=iterations)       
            
    '''
    \brief Removes the small noises from a image
    \note The image need to be a 1 channel image
    \raises AssertionError in case the provided image is not single channel or is not binary
    \param[in] img Numpy image to be processed
    \param[in] kernel_size The size of the kernel used. if set to x the kernel size will be (x, x)
    \param[in] iterations Number of iterations of the convolution
    \returns Processed image
    '''
    @staticmethod
    def removeNoise(img, kernel_size=3, iterations=1):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accepted')
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel, iterations=iterations)

    '''
    \brief Fills small holes in regions smaller than the kernel
    \note The image need to be a 1 channel image
    \raises AssertionError in case the provided image is not single channel or is not binary
    \param[in] img Numpy image to be processed
    \param[in] kernel_size The size of the kernel used. if set to x the kernel size will be (x, x)
    \param[in] iterations Number of iterations of the convolution
    \returns Processed image
    '''
    @staticmethod
    def fillHoles(img, kernel_size=3, iterations=1):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accept')
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel, iterations=iterations)


    '''
    \brief Binarizes the image by the provided threshold
    \note The image need to be single channel image
    \raises AssertionError in case the provided image is not single channel
    \param[in] img Numpy image to be processed
    \param[in] threshold Threshold value used for the binarization
    \returns Processed image
    '''
    @staticmethod
    def binarization(img, threshold=128):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accept')
        ret_val, bin_img = cv2.threshold(img, threshold, maxval=255, type=cv2.THRESH_BINARY)
        return bin_img


    '''
    \brief Do the full post processing with default values. For convenience
    \note The image need to be a 1 channel image
    \raises AssertionError in case the provided image is not single channel or is not binary
    \param[in] img Numpy image to be processed
    \returns Processed image
    '''
    @staticmethod
    def fullPostProcessing(img):
        if(len(img.shape)!=2):
            raise AssertionError('Only single channel images are accept')
        bin_img = MaskPostProcessing.binarization(img)
        bin_img = MaskPostProcessing.removeNoise(bin_img)
        bin_img = MaskPostProcessing.removeClumps(bin_img)
        bin_img = MaskPostProcessing.dilateImage(bin_img)
        return bin_img


    '''
    \brief Find contours (filled regions) in an image with area within lower and upper tolerance, then return bounding boxes and a mask for these.
    \param[in] img Numpy image to be processed. .astype("uint8")
    \param[in] lower_area Lower boundary of required contour area (unit: pixels)
    \param[in] upper_area Upper boundary of required contour area (unit: pixels)
    \param[in] mode cv2.findContour RetrievalModes. Default is to return all contours ( cv2.RETR_LIST ) but also can return only outermost contours (e.g. cv2.RETR_EXTERNAL) alternative options are cv2.RETR_CCOMP or cv2.RETR_TREE (see opencv docs).
    \returns foreground A greyscale mask of the countours within the area.
    \returns boxes Coordinates (x,y,w,h) of the bounding boxes of each contour. If no boxes of sufficient size, return empty list.
    '''
    @staticmethod
    def findContours(img, lower_area=150, upper_area=100 * 100, mode=cv2.CHAIN_APPROX_SIMPLE):
        # img = MaskPostProcessing.binarization(img)

        # Initialise the "foreground" grey mask as empty. For plotting in imshow().
        foreground = np.zeros_like(img)

        contours, hierarchy = cv2.findContours(img,
                                        mode,
                                        cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i, cnt in enumerate(contours):

            if lower_area < cv2.contourArea(cnt) < upper_area:

                (x, y, w, h) = cv2.boundingRect(cnt)
                
                if (w > 0) and (h > 0):
                    boxes.append((x, y, w, h))

                # Create a mask image that contains the contour filled in with a
                # large enough area
                cv2.drawContours(foreground, contours, i, color=255, thickness=-1)

        return foreground, boxes
    
    '''
    \brief Simple wrapper function to retrn contours for a mask.
    \param[in] mask Greyscale 8-bit (.astype("uint8")) numpy image to be processed.
    \returns contours Detected contours. Each contour is stored as a vector of points.
    '''
    @staticmethod
    def getContours(mask):
        contours, hierarchy = cv2.findContours(mask,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
        return contours


    """
    \brief  Applied a bitwise mask to remove white areas from a greyscale image
    removing all areas inside rectangles of boundingbox_file
    \param[in] img Numpy greyscale image to be processed
    \param[in] unwanted_boxes list of boxes in (x,y,w,h) format to be removed from the input mask.
    \returns masked_img A greyscale mask with regions inside unwanted_boxes removed (black).
    """
    @staticmethod
    def removeUnwantedBoxes(img, unwanted_boxes):

        frame_height, frame_width = img.shape

        unwanted_areas = np.zeros((frame_height, frame_width), np.uint8) #Create the arbitrary input img

        #in place drawing of unwanted areas
        [cv2.rectangle(unwanted_areas, (x, y), (x + w, y + h),  (255, 255, 255), thickness=cv2.FILLED)
            for (x, y, w, h) in unwanted_boxes]

        mask_inv = cv2.bitwise_not(unwanted_areas)

        masked_img=cv2.bitwise_and(mask_inv,img.astype("uint8"), mask=mask_inv)

        return masked_img

    """
    \brief  By default, applies a bitwise mask to remove white areas of the provided `unwanted_areas_mask` from white regions of the provided greyscale image. Essentially applying a mask to a mask. If inverse flag is set to False, it will do remove black areas `unwanted_areas_mask` from white regions of the provided greyscale image.
    removing all areas inside rectangles of boundingbox_file
    \param[in] img Numpy greyscale image to be processed
    \param[in] mask A greyscale numpy image array .astype("np.uint8")
    \param[in] inverse boolean. If set to False, it will use black regions from the mask as regions to remove from the image. Default is True, meaning use white regions of the mask to remove from the image.
    \returns masked_img A greyscale mask with regions inside unwanted_boxes removed (black).
    """
    @staticmethod
    def removeUnwantedRegions(img, unwanted_areas_mask, inverse=True):

        frame_height, frame_width = img.shape

        if inverse:
            unwanted_areas_mask = cv2.bitwise_not(unwanted_areas_mask)

        masked_img=cv2.bitwise_and(unwanted_areas_mask,img.astype("uint8"), mask=unwanted_areas_mask)

        return masked_img

    '''
    \brief  Takes a fgmask and applies a named set of morphological filters  
    \param[in] filtername A string corresponding to a filter set
    \param[in] fgmask A foreground mask numpy array
    \returns fgmask The modified foreground mask.
    '''
    @staticmethod
    def apply_filter(filtername, fgmask):
             
        if filtername == 'filter1_high_recall_high_noise':
             fgmask = MaskPostProcessing.fillHoles(fgmask, kernel_size=3, iterations=2)
             fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=5, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
             
             #unwanted_AOI_boxes = bbox_utils.getBoxesFromFile(os.path.join('../../', Config.masks['AOI_IGNORE_BOXES']),
             #                                                 (Config.video["SCALE_FACTOR1_FRAME_W"], Config.video["SCALE_FACTOR1_FRAME_H"]))
             #from LowLevelModules.VisualizationTools import VisualizationTools as visualize 
             #visualize.drawBoxes(fgmask, unwanted_AOI_boxes)
             #cv2.imshow('debug', fgmask)
             #fgmask = MaskPostProcessing.removeUnwantedBoxes(fgmask, unwanted_AOI_boxes)

        if filtername == 'filter2_low_recall_low_noise':
            fgmask = MaskPostProcessing.removeNoise(fgmask, kernel_size=2, iterations=1)
            fgmask = MaskPostProcessing.fillHoles(fgmask, kernel_size=3, iterations=2)
            fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=5, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
            fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=8, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
            fgmask = MaskPostProcessing.fillHoles(fgmask, kernel_size=8, iterations=1)
            
        if filtername == 'filter3_low_recall_low_noise_entrance':
            # enhanced dilation to stop people breaking up which is highly problematic for people's tracks crossing the line
            fgmask = MaskPostProcessing.removeNoise(fgmask, kernel_size=2, iterations=1)
            fgmask = MaskPostProcessing.fillHoles(fgmask, kernel_size=3, iterations=2)
            fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=5, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
            fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=8, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
            fgmask = MaskPostProcessing.dilateImageVertically(fgmask, kernel_size=8, iterations=2, clip_edge_columns=1) # humans are often shaped like y-stretched ovals
            fgmask = MaskPostProcessing.fillHoles(fgmask, kernel_size=8, iterations=1)

        fgmask = MaskPostProcessing.binarization(fgmask)
        
        return fgmask
    
'''
\brief Makes a vertical shaped elipse structuring element, for use with opencv morphological filters. 
\param[in] kernel_size The size of the kernel used. if set to x the kernel size will be (x, x)
\param[in] clip_edge_columns Number of iterations of the convolution
\returns structuring_element (kernel_size, kernel_size) numpy array 
'''
def customVerticalDilationKernel(kernel_size, clip_edge_columns):
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # clip the edges by clip_edge_columns to make a vertical shape
    for i in range(clip_edge_columns):
        structuring_element[:,i]= 0
        structuring_element[:,-(i+1)] = 0
    return structuring_element 