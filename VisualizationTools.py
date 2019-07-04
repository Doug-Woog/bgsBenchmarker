'''
\brief Tools to help plot results of analysis in OpenCV. e.g. drawing bounding boxes onto a frame.
'''

import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image # For Japanese Text overlays

import sys
sys.path.insert(0, "../../src")


color_dictionary = {"green": (10, 255, 10), "blue": (255, 10, 10), "red" : (10, 10, 255), "white" : (255,255,255)}



class VisualizationTools(object):
    '''
    \brief Draws bounding boxes onto the img in place
    \param[in,out] img Numpy image array to be drawn onto
    \param[in] boxes list of boxes to draw in (x, y, w, h) format
    \param[in] color Optional color string in color_dictionary (blue/green/red) or a RGB color tuple
    '''
    @staticmethod
    def drawBoxes(img, boxes, color="green"):

        if isinstance(color, str):
            color_tuple=color_dictionary[color.lower()]
        else:
            color_tuple=color

        [cv2.rectangle(img, (x, y), (x + w, y + h), color_tuple, 2) for (x, y, w, h) in boxes]

    '''
    \brief Draws bounding box onto the img in place
    \param[in,out] img Numpy image array to be drawn onto
    \param[in] boxes Single box to draw in (x, y, w, h) format
    \param[in] color Optional color string in color_dictionary (blue/green/red) or a RGB color tuple
    '''
    @staticmethod
    def drawBox(img, box, color="green"):
        (x, y, w, h) = box

        if isinstance(color, str):
            color_tuple=color_dictionary[color.lower()]
        else:
            color_tuple=color

        cv2.rectangle(img, (x, y), (x + w, y + h), color_tuple, 2)


    '''
    \brief Given the 16 grid camera layout provide a selection of: cameras 1 to 16 (integer) or string "3x3"/"4x4" to crop to this
    \param[in] img The 16 grid camera feed
    \param[in] A string or integer representing the selection. ("3x3"/"4x4" or integer 1-16 for each camera.
    \param[in] Scale factor, the default 1 is for when the frame is the same size as default (h=1080, w=1440). If the frame if half size, this value should be 0.5.
    \returns img The cropped image.
    \deprecated use CameraManager class instead
    '''
    @staticmethod
    def cropToCamera(img, selection, scale_factor=1):

            (x,y), (x2, y2), vid_height, vid_width = VisualizationTools.getCameraFrameDimensions(selection, scale_factor)

            img = img[y:y+vid_height, x:x+vid_width]

            return img

    '''
    \brief Given the 16 grid camera layout provide a selection of: cameras 1 to 16 (integer) or string "3x3"/"4x4" to crop to this
    \param[in] img The 16 grid camera feed
    \param[in] A string or integer representing the selection. ("3x3"/"4x4" or integer 1-16 for each camera.
    \returns img The cropped image.
    \deprecated For the old footage which is 240 pixels shifted from the current footage.
    '''
    @staticmethod
    def old_cropToCamera(img, selection):
            cam_h=270
            cam_w=360

            left_offset=240

            if selection in [1,2,3,4]:
                #Camera 1 or 2 or 3 or 4
                y=0
                x=left_offset + (selection-1)*cam_w

            if selection in [5,6,7,8]:
                y=cam_h
                x=left_offset + (selection-5)*cam_w

            if selection in [9,10,11,12]:
                y=cam_h*2
                x=left_offset + (selection-9)*cam_w

            if selection in [13,14,15,16]:
                y=cam_h*3
                x=left_offset + (selection-13)*cam_w

            vid_height=cam_h
            vid_width=cam_w

            if selection == "3x3":
                #3x3 top-left box.
                y=0
                vid_height=cam_h*3
                x=left_offset
                vid_width=cam_w*3

            if selection == "4x4":
                y=0

            img = img[y:y+vid_height, x:x+vid_width]

            return img


    '''
    \brief Put black text on image with a transparent white background around the text. The text can be Japanese, unlike default OpenCV text function.
    \param[in] text String to be written onto the frame
    \param[in] img numpy image array to be drawn onto. Not changed in place.
    \param[in] text_offset_x Leftmost X Position of the text
    \param[in] text_offset_y Topmost Y Position of the text
    \returns img image with translucent box filled with text. Input image not changed in place.
    '''
    @staticmethod
    def addOverlayText(text, img, text_offset_x, text_offset_y, color=(0,0,0), fontsize=18):
        text_color=color
        fontpath = "./assets/JKG-L_3.ttf"
        if os.path.isfile(fontpath):
            pass
        else:
            raise FileNotFoundError(f"The font file {fontpath} was not found")

        font = ImageFont.truetype(fontpath, fontsize)

        text_size = font.getsize(text)

        img_pil = Image.fromarray(img) # Convert array to 8 bit integer type (0 to 255) PIL Image.

        img_pil = img_pil.convert("RGBA")

        tmp = Image.new('RGBA', img_pil.size, (0,0,0,0))

        draw_tmp = ImageDraw.Draw(tmp)

        box_xy = [text_offset_x,text_offset_y,text_offset_x+text_size[0], text_offset_y+text_size[1]]

        alpha = 150

        draw_tmp.rectangle(box_xy, fill=(255,255,255,alpha))

        draw_tmp.text((text_offset_x, text_offset_y), text, font = font , fill = text_color )

        # Alpha composite the two images together.
        img_pil = Image.alpha_composite(img_pil, tmp)

        img_pil = img_pil.convert("RGB") # Remove alpha for saving in jpg format.

        img = np.array(img_pil) # Convert PIL back to array

        return img

    '''
    \brief Takes a pair of greyscale or color image, if greyscale it converts it color, then mixes them into a blended image.
    \param[in] img1 A numpy image array in greyscale (2-channel) or color (3-channel).
    \param[in] img2 A numpy image array in greyscale (2-channel) or color (3-channel).
    \param[in] alpha A float between 0 and 1 representing how much of img1 versus img2 should be shown in the output image. Default is 0.5, meaning half-half.
    \returns mixed_img An overlay of img1 and img2 blended according to alpha value.
    '''
    @staticmethod
    def blendImages(img1,img2, alpha=0.5):
        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        beta = (1.0 - alpha)
        mixed_img = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
        return mixed_img

    '''
    \brief Draws a solid, dashed or dotted line.
    \param[in,out] img Image to be written to (in place).
    \param[in] pt1 An (x,y) tuple for the line start coordinate.
    \param[in] pt2 An (x,y) tuple for the line end coordinate.
    \param[in] color opencv color BGR tuple or a string in the color dictionary of this class.
    \param[in] thickness line thickness
    \param[in] style string of "normal", "dotted", or "dashed". Default "normal".
    \param[in] gap For dashed/dotted line, spacing between elements
    '''
    @staticmethod
    def drawLine(img,pt1,pt2,color,thickness=1,style='normal',gap=10):
        if isinstance(color, str):
            color=color_dictionary[color.lower()]
        else:
            pass

        dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
        pts= []
        for i in  np.arange(0,dist,gap):
            r=i/dist
            x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
            y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
            p = (x,y)
            pts.append(p)

        if style == "normal":
            cv2.line(img, pt1, pt2, color, thickness)

        if style=='dotted':
            for p in pts:
                cv2.circle(img,p,thickness,color,-1)
        else:
            s=pts[0]
            e=pts[0]
            i=0
            for p in pts:
                s=e
                e=p
                if i%2==1:
                    cv2.line(img,s,e,color,thickness)
                i+=1



    '''
    \brief Put `cropped_img` is overlaid ontop of `img` at the appropriate position for camera index specified in `selection`. If the frame has been resized, the appropriate `scale_factor` must be provided.
    \param[in] selection A string or integer representing the selection. ("3x3"/"4x4" or integer 1-16 for each camera.
    \param[in] Scale factor, the default 1 is for when the frame is the same size as default (h=1080, w=1440). If the frame if half size, this value should be 0.5.
    \returns img The original `img` with `cropped_img` overlaid at the correct position.
    '''
    @staticmethod
    def overlayCroppedImgToFullImage(img, selection, cropped_img, scale_factor):

        (x,y), (x2, y2), vid_height, vid_width = VisualizationTools.getCameraFrameDimensions(selection, scale_factor)

        img[y:y+vid_height, x:x+vid_width] = cropped_img

        return img

        '''
    \brief Given the 16 grid camera layout provide a selection of: cameras 1 to 16 (integer) or string "3x3"/"4x4" provide information about the the cropped image dimensions.
    \param[in] selection A string or integer representing the selection. ("3x3"/"4x4" or integer 1-16 for each camera.
    \param[in] Scale factor, the default 1 is for when the frame is the same size as default (h=1080, w=1440). If the frame if half size, this value should be 0.5.
    \returns (x,y), (x+cam_w, y+cam_h), vid_height, vid_width The parameters for this cropped frame.
    \deprecated Use CameraManager
    '''
    @staticmethod
    def getCameraFrameDimensions(selection, scale_factor=1):
        cam_h=int(270*scale_factor)
        cam_w=int(360*scale_factor)
        left_offset=0

        if selection in [1,2,3,4]:
            #Camera 1 or 2 or 3 or 4
            y=0
            x=left_offset + (selection-1)*cam_w

        if selection in [5,6,7,8]:
            y=cam_h
            x=left_offset + (selection-5)*cam_w

        if selection in [9,10,11,12]:
            y=cam_h*2
            x=left_offset + (selection-9)*cam_w

        if selection in [13,14,15,16]:
            y=cam_h*3
            x=left_offset + (selection-13)*cam_w

        vid_height=cam_h
        vid_width=cam_w

        if selection == "3x3":
            #3x3 top-left box.
            y=0
            vid_height=cam_h*3
            x=left_offset
            vid_width=cam_w*3

        if selection == "4x4":
            y=0

        return (x,y), (x+cam_w, y+cam_h), vid_height, vid_width



    """
    \brief Put text into a image in place. Faster than addOverlayText function.
    \param[in,out] im An 3-channels image
    \param[in] text The text will be drawn in the image
    \param[in] rect The bounding box for the text. Only first two coordinates x and y used.
    \param[in] color Tuple containing 3 values represent to the color
    \return The image with drawn text
    """
    @staticmethod
    def putText(im, text, rect, color=(255, 255, 255)):
        im = cv2.putText(im, text,
                        (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
        return im

