import numpy as np
import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, startY, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        #pad_w, pad_h = int(0.15*w), int(0.05*h)
        #   cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

        #without padding
        cv2.rectangle(img, (x, y + startY), (x+w, y+ startY+h), (0, 255, 0), thickness)

def save_detections(img, rects):
    count = 0
    for x, y, w, h in rects:
        cv2.imwrite("detections/" + str(count) + ".png", img[y:y+h,x: x+w])
        count += 1


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #read the image
    uncropped = cv2.imread("test.png")
    #define the upper band where not to look
    startY = 125
    #crop the frame to prevent detection before startY
    frame = uncropped[startY:720, 0: 1280]


    found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

    #save detections from cropped img
    save_detections(frame,found)
    
    #draw detections on uncropped img
    draw_detections(uncropped, startY, found)
    
    #image = cv2.resize(frame, (1280,720))
    cv2.imwrite("out.png",uncropped)
