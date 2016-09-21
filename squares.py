#!/usr/bin/env python
#Sources: http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

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

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0) #Blurs the image using the Gaussian filter
    squares = []
    for gray in cv2.split(img): #Splits the images into seperate color channels
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5) #Finds edges in an images using the Canny algorith
                bin = cv2.dilate(bin, None) #Dilates an images by using a specific structuring element
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #Approximates a polygonal curve with the specified precision
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


'''
def find_circles(img):
    #img = #Do I re-GaussianBlur the image? I don't think I need to since it's done in find_squares(img)
    img_gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    ks = 31
    gb_kernel = cv2.getGaborKernel((ks,ks),4.0,np.pi,10.0,0.5,0,cv2.CV_32F)
    img_filtered = cv2.filter2D(gray_img, cv2.CV_8U, gb_kernel.transpose())
    circles = cv2.HoughCircles(img_filtered, cv2.HOUGH_GRADIENT, 1.2, 100)
    return circles

#        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 200)

def find_circles(img):
    img = cv2.GaussianBlur(img, (5, 5), 0) #Blurs the image using the Gaussian filter
    circles = []
    for gray in cv2.split(img): #Splits the images into seperate color channels
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    return circles
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5) #Finds edges in an images using the Canny algorith
                bin = cv2.dilate(bin, None) #Dilates an images by using a specific structuring element
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #Approximates a polygonal curve with the specified precision
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)'''



if __name__ == '__main__':
    from glob import glob
    for fn in glob('../data/pic*.png'):
        img = cv2.imread(fn)
        squares = find_squares(img)
        cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )
        cv2.imshow('squares', img)
#	circles = find_circles(img) #Added by Cristian
#        cv2.drawContours( img, circles, -1, (0, 255, 0), 3 ) #Added by Cristian
#        cv2.imshow('circles', img) #Added by Cristian
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()
