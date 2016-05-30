'''
SkinDetector
Takes an active webcam or pre-saved video file, and produces split-screen video
showing the original and a skin-only video.

Last Updated: 2016-May-29
First Created: 2016-May-29

Python 2.7
Chris

http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
'''

import imutils
import numpy as np
import cv2

def skin_detector(video=False):
    '''
    Detects skin from video, flag False uses the user's own webcam.
    Else if recorded video is provided (e.g example.mov) then that is used.
    '''

    # define the upper and lower boundaries of the HSV (!!!) pixel
    # intensities to be considered 'skin'

    lower = np.array([0, 48, 80], dtype = 'uint8')
    upper = np.array([20, 255, 255], dtype = 'uint8')

    if not video:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(video)

    # loop over frames in the video
    while True:
        # grab current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video

        if video and not grabbed:
            break

        # resize the frame, convert to HSV color space
        # determine the HSV pixel intensities that fall into
        # specified upper and lower boundaries

        frame = imutils.resize(frame, width = 400)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask to help remove noise, then apply the mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        # show the skin in the image along with the mask
        cv2.imshow('images', np.hstack([frame, skin]))

        # stop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup and close
    camera.release()
    cv2.destroyAllWindows()

def save_skin_detector(output_filename, video_file=0):
    '''
    Saves video produced by the skin detector function to a file.
    Defaults to webcam, else if file name is provided uses that.
    Outputs to specified filename.

    (have only been able to get to work with webcam so far)
    '''

    cap = cv2.VideoCapture(video_file)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#skin_detector('test.mov')
#skin_detector()

save_skin_detector('test2.mp4', 'bla.mp4')
