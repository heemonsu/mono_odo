import numpy as np
import cv2

from scipy.ndimage import rotate
from scipy.ndimage import zoom

import sys

#path to the image
filename = sys.argv[1]
#where to save the image
destination = sys.argv[2]




def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out



#load an image
img = cv2.imread(filename,0)
rows,cols = img.shape

#rotate the image
rot = rotate(img, 30, reshape=False)
new_path = destination + "rotation.png"
cv2.imwrite(new_path, rot)
#cv2.imshow('image',rot)
#cv2.waitKey(0)


#zoom-in on an image
zm1 = clipped_zoom(img, 2)
new_path = destination + "zoonin.png"
cv2.imwrite(new_path, zm1)
#cv2.imshow('image',zm1)
#cv2.waitKey(0)

#zoom-out on an image
zm2 = clipped_zoom(img, 0.5)
new_path = destination + "zoomout.png"
cv2.imwrite(new_path, zm2)
#cv2.imshow('image',zm2)
#cv2.waitKey(0)

#translation of the image
M = np.float32([[1,0,100],[0,1,50]])
tran = cv2.warpAffine(img,M,(cols,rows))
new_path = destination + "translation.png"
cv2.imwrite(new_path, tran)
#cv2.imshow('image', tran)
#cv2.waitKey(0)

#blurring
kernel = np.ones((5,5),np.float32)/25
blurr = cv2.filter2D(img,-1,kernel)
new_path = destination + "blurr.png"
cv2.imwrite(new_path, blurr)
#cv2.imshow('image', blurr)
#cv2.waitKey(0)

#rotation+zoom
zm_ = clipped_zoom(img, 1.5)
rot_ = rotate(zm_, 30, reshape=False)
new_path = destination + "rotzoom.png"
cv2.imwrite(new_path, rot_)
#cv2.imshow('image', rot_)
#cv2.waitKey(0)


