# USAGE
# python test_video.py --video ../videos/WarriorsOffense.mp4

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True,
                help="path to input video")
args = vars(ap.parse_args())


def predict_label(image, playerModel, playerLB):
    # load the image
    output = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image using Keras' multi-output functionality
    playerProba = playerModel.predict(image)

    # find indexes of the player outputs with the
    # largest probabilities, then determine the corresponding class label
    idx = playerProba[0].argmax()
    label = playerLB.classes_[idx]
    score = playerProba[0][idx] * 100

    return (label, score)


def draw_detections(img, boxes, playerNotPlayermodel, playerModel, playerLB):
    for box in boxes:
        x = box[1]
        y = box[0]
        x2 = box[3]
        y2 = box[2]
        cropped = img[y:y2, x: x2]
        image = img[y:y2, x: x2]

        if x2-x < 1280/4 and y2-y < 720 / 2:
            # pre-process the image for classification
            image = cv2.resize(image, (28, 28))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            (notPlayer, player) = playerNotPlayermodel.predict(image)[0]
            if player > notPlayer:
                (label, score) = predict_label(cropped, playerModel, playerLB)
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1]*im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":

    detector_model_path = '../PlayerNotPlayer/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=detector_model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(args['video'])
    count = 0

    # load the trained playerNotPlayer convolutional neural network
    print("[INFO] loading network...")
    playerNotPlayerModel = load_model(
        "../PlayerNotPlayer/player_not_player.model")

    # load the trained player convolutional neural network, followed
    # by the player label binarizer
    print("[INFO] loading network...")
    playerModel = load_model("player.model", custom_objects={"tf": tf})
    playerLB = pickle.loads(open("player_lb.pickle", "rb").read())

    while True:
        players = []
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                players.append(boxes[i])

        draw_detections(img, players, playerNotPlayerModel,
                        playerModel, playerLB)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
