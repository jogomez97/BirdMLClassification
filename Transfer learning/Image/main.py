# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import os
import json

# construct the argument parse and parse the arguments

args = {
    "dataset": '../Imatge/02_Base de dades/full',
    "model": 'image.h5'
}

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=200)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


baseModel = VGG16(weights="imagenet", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


print("[INFO] training head...")
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=25, verbose=1)

model.save("warmup.h5")
with open('history_warmup.txt', 'w') as outfile:
    json.dump(history.history, outfile)
    outfile.close()

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# checkpoint
filepath="./model_checkpoint/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("[INFO] fine-tuning model...")
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=100, verbose=1, callbacks=callbacks_list)
# save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])
with open('history_train.txt', 'w') as outfile:
    json.dump(history.history, outfile)
    outfile.close()

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))
