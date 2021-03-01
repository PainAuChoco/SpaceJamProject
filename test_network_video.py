# USAGE
# python test_network_video.py --model player_not_player.model --video WarriorsOffense.mp4

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
args = vars(ap.parse_args())


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, model):
    for x, y, w, h in rects:
        image = img[y:y+h,x: x+w]
        
        # pre-process the image for classification
        image = cv2.resize(image, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        (notPlayer, player) = model.predict(image)[0]
        if(player > notPlayer): 
            c = (0,255,0)
        else :
            c = (0,0,255)

        cv2.rectangle(img, (x, y), (x+w, y+h), c, 2)

if __name__ == '__main__':

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])

    # load the descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    cap=cv2.VideoCapture(args["video"])
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter("output.avi", fourcc, 29.97, (1280,720))

    while True:
        _,frame=cap.read()
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame,found, model)
        cv2.imshow('feed',frame)

        image = cv2.resize(frame, (1280,720))
        out.write(image)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
    out.release()
