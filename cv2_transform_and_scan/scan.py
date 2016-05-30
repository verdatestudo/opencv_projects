'''
Scan
Scans a receipt and produces a top-down view of the image.

Last Updated: 2016-May-29
First Created: 2016-May-29
Python 2.7
Chris

http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
'''

from transform import four_point_transform
from skimage.filters import threshold_adaptive

import numpy as np
import imutils
import cv2

def scan_receipt(image_file):
    '''
    Takes a scanned receipt and returns a top-down scan view of the image.
    '''

    img = cv2.imread(image_file)

    # resize to 500 height, store ratio of original
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = imutils.resize(img, height = 500)

    # grayscale, blur, find edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 25, 200) # for sainsbury receipt
    #edged = cv2.Canny(gray, 75, 200) # for scan.png

    # show images
    print 'Step 1: Edge Detection'
    cv2.imshow('image', img)
    cv2.imshow('gray', gray)
    cv2.imshow('edged', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the contours and get the largest ones
    #_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if contour has four points, then it's probably our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    print 'Step 2: Find contours of paper'
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow('outline', img)
    cv2.waitKey(0)

    # apply the four point transformation to obtain top-down view of image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert to grayscale, threshold to get black and white paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 251, offset = 10)
    warped = warped.astype('uint8') * 255

    # show the original and scanned image
    print 'Step 3: Apply perspective transform'
    cv2.imshow('Original', imutils.resize(orig, height = 650))
    cv2.imshow('Scanned', imutils.resize(warped, height = 650))
    cv2.waitKey(0)

image_file = 'test.jpg'

scan_receipt(image_file)
