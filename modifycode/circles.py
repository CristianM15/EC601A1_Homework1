#!/usr/bin/env python
#Sources: http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
#Sources: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

if __name__ == '__main__':
    from glob import glob
    for fn in glob('../data/pic*.png'):
        img = cv2.imread(fn,0)
        img = cv2.medianBlur(img,5)
#        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ,img)
    #    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR,cimg)
#        gb_kernel = cv2.getGaborKernel((31,31),4.0,np.pi,10.0,0.5,0,cv2.CV_32F)
#        img = cv2.filter2D(img, cv2.CV_8U, gb_kernel)
 # Determine image depth here
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        #cv2.imshow('detected circles',cimg)
        cv2.imshow('detected circles',img)
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()
