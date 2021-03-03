# USAGE
# python test_network_photo.py --model player_not_player.model --photo test.png

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-v", "--photo", required=True,
	help="path to input photo")
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
        cv2.rectangle(img, (x, y), (x+w, y+ h), (0, 255, 0), thickness)

def predictPlayers(img, rects, model):
    for x, y, w, h in rects:
        image = img[y:y+h,x: x+w]
        
        # pre-process the image for classification
        image = cv2.resize(image, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        (notPlayer, player) = model.predict(image)[0]

        # build the label
        label = "Player" if player > notPlayer else "Not Player"
        proba = player if player > notPlayer else notPlayer
        label = "{}: {:.2f}%".format(label, proba * 100)

        cv2.putText(img, label, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

if __name__ == '__main__':

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #read the image
    frame = cv2.imread(args["photo"])

    found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

    predictPlayers(frame,found, model)
    
    #draw detections on uncropped img
    draw_detections(frame, found)
    
    #image = cv2.resize(frame, (1280,720))
    cv2.imwrite("out.png",frame)
