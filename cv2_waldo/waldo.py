'''
Waldo

Takes a 'Where's Waldo' puzzle and finds Waldo.

Last Updated: 2016-Jun-03
First Crated: 2016-Jun-03

Python 2.7
Chris

http://www.pyimagesearch.com/wp-content/uploads/2014/11/opencv_crash_course_waldo.pdf
'''

import numpy as np
import imutils
import cv2

def find_waldo(puzzle_image, waldo_image):
    '''
    Takes a puzzle image and an image of waldo, and returns his location in the puzzle.
    '''
    puzzle = cv2.imread(puzzle_image)
    waldo = cv2.imread(waldo_image)
    (waldo_height, waldo_width) = waldo.shape[:2]

    # find waldo. TMCC0EFF is the matching method used.
    result = cv2.matchTemplate(puzzle, waldo,  cv2.TM_CCOEFF)
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

    top_left = maxLoc
    bot_right = (top_left[0] + waldo_width, top_left[1] + waldo_height)
    roi = puzzle[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]

    # construct a darkened transparent layer for everything except waldo.
    mask = np.zeros(puzzle.shape, dtype = 'uint8')
    puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

    # put original waldo in to look brighter than rest of image
    puzzle[top_left[1]:bot_right[1], top_left[0]:bot_right[0]] = roi

    cv2.imshow('Puzzle', imutils.resize(puzzle, height = 650))
    cv2.imshow('Waldo', waldo)
    cv2.waitKey(0)


find_waldo('waldo_puzzle1.png', 'waldo.png')
