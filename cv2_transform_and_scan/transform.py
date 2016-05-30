'''
Transform
Perspective transforms an image

Last Updated: 2016-May-29
First Created: 2016-May-29
Python 2.7
Chris

http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
'''

import numpy as np
import cv2

def order_points(pts):
    '''
    Takes a list of four points specifying the (x, y) co-ordinates of each point of the rectangle.
    Returns an ordered list of the four points (top-left, top-right, bot-right, bot-left)
    '''
    # init coords such as entry is top-left, top-right, bot-right, bot-left
    rect = np.zeros((4, 2), dtype = 'float32')

    # top-left smallest sum, bot-right largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now compute the difference between the points: top-right = largest, bot-left = smallest

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    '''
    Takes an image and four cords and returns a transformed image.
    '''
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width and height of the new image, which will be the max distance
    # between suitable co-ordinates

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],\
    [0, maxHeight - 1]], dtype = 'float32')

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def transform_example(image_file, cords):
    '''
    An example implementation of transform.
    '''
    img = cv2.imread(image_file)
    pts = np.array(cords, dtype = 'float32')

    warped = four_point_transform(img, pts)

    cv2.imshow('Original', img)
    cv2.imshow('Warped', warped)
    cv2.waitKey(0)

#transform_example('transform_example_1.jpg', [(75, 255), (378, 130), (505, 280), (200, 470)])
