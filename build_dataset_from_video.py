# USAGE
# python build_dataset_from_video.py --video NetsOffense.mp4 --output detections

import numpy as np
import cv2
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video")
ap.add_argument("-o", "--output", required=True,
	help="path to ouptut folder")
args = vars(ap.parse_args())


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        #pad_w, pad_h = int(0.15*w), int(0.05*h)
        #   cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

        #without padding
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

def save_detections(img, rects, count):
    for x, y, w, h in rects:
        cv2.imwrite(args["output"] + "/" + str(count) + ".png", img[y:y+h,x: x+w])
        count += 1
    return count

if __name__ == '__main__':
    VIDEO_FPS = 30

    cap=cv2.VideoCapture(args["video"])

    #number of images per second we want, 30 gives 19,470 pics for 54 seconds
    wanted_fps = 3, 
    rate = 30

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    
    count = 0
    tmp = 0
    while True:
        
            _,frame=cap.read()
            if(tmp == rate):
                found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
                count = save_detections(frame, found, count)
                tmp = 0
            else :
                tmp += 1

    cv2.destroyAllWindows()
