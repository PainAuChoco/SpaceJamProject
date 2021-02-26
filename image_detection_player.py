import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np


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
    model = load_model("player_not_player.model")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #read the image
    uncropped = cv2.imread("test.png")
    #define the upper band where not to look
    startY = 0
    #crop the frame to prevent detection before startY
    frame = uncropped[startY:720, 0: 1280]

    found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

    predictPlayers(frame,found, model)

    #save detections from cropped img
    save_detections(frame,found)
    
    #draw detections on uncropped img
    draw_detections(uncropped, startY, found)
    
    #image = cv2.resize(frame, (1280,720))
    cv2.imwrite("out.png",uncropped)
