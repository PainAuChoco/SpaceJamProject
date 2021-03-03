# USAGE
# python train.py --dataset train --model player.model --playerbin output/player_lb.pickle 

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.playernet import PlayerNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 8
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("train")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data
data = []
playerLabels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = img_to_array(image)
	data.append(image)

	# extract the player name from the path and update the respective lists
	player = imagePath.split(os.path.sep)[-2]
	playerLabels.append(player)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label list to NumPy array prior to binarization
playerLabels = np.array(playerLabels)

# binarize the label
print("[INFO] binarizing labels...")
playerLB = LabelBinarizer()
playerLabels = playerLB.fit_transform(playerLabels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, playerLabels, test_size=0.2, random_state=42)
(trainX, testX, trainY, testY) = split

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest"
	)

# initialize our PlayerNet multi-output network
model = PlayerNet.build(96, 96,
	numPlayers=len(playerLB.classes_),
	finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"player_output": "categorical_crossentropy",
}
lossWeights = {"player_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX,{"player_output": testY}),
	epochs=EPOCHS,
	verbose=1
	)

# save the model to disk
print("[INFO] serializing network...")
model.save("player.model", save_format="h5")

# save the player binarizer to disk
print("[INFO] serializing player label binarizer...")
f = open("player_lb.pickle", "wb")
f.write(pickle.dumps(playerLB))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")