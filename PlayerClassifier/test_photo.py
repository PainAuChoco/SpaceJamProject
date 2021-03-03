# USAGE
# python test_photo.py --image test/1.png

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network from disk, followed
# by the player label binarizer
print("[INFO] loading network...")
model = load_model("player.model", custom_objects={"tf": tf})
playerLB = pickle.loads(open("player_lb.pickle", "rb").read())

# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
playerProba = model.predict(image)

# find indexes of the player outputs with the
# largest probabilities, then determine the corresponding class label
idx = playerProba[0].argmax()
label = playerLB.classes_[idx]
score = playerProba[0][idx] * 100

# draw the label on the image
text = "player: {} ({:.2f}%)".format(label,score)
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print()

#display the other scores in the console
for i in range(0, len(playerLB.classes_)-1):
	label = playerLB.classes_[i]
	score = playerProba[0][i] * 100

	# draw the label on the image
	text = str(i) + ": {} ({:.2f}%)".format(label,score)

	# display the predictions to the terminal as well
	print("[SCORES] {}".format(text))

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)