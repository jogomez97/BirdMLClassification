import sys 
sys.path.append('D:\La Salle\Xavier Sevillano Domínguez - TFM Juan Gómez\Transfer learning')


# import
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import os
import json

# custom imports
from custom_dataset_loader_and_splitter import CustomDatasetLoaderAndSplitter
from fcheadnet import FCHeadNet

# construct the argument parse and parse the arguments
args = {
    "dataset": 'D:/La Salle\Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s', 
    "model": 'output.h5'
}
batch_size = 32

# load the images and get the splits
classNames = list(os.listdir(args["dataset"]))
c = CustomDatasetLoaderAndSplitter(args["dataset"], validation=0.2, test=0.1)
trainX, trainY, valX, valY, testX, testY = c.load_and_split()

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
valY = LabelBinarizer().fit_transform(valY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 network, ensuring the head FC layer sets are left off (include_top=False)
print("[INFO] loading model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
# typical warmup are 10-30 epoch
print("[INFO] training head...")
history = model.fit(trainX, trainY, validation_data=(valX, valY), 
        epochs=25, batch_size=batch_size
        ,verbose=1)

model.save("warmup.h5")
with open('history_warmup.txt', 'w') as outfile:
    json.dump(history.history, outfile)
    outfile.close()

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# checkpoint
filepath="./model_checkpoint/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
print("[INFO] fine-tuning model...")
history = model.fit(trainX, trainY, validation_data=(valX, valY), 
        epochs=150, batch_size=batch_size
        ,verbose=1, callbacks=callbacks_list)

model.save(args["model"])
with open('history_training.txt', 'w') as outfile:
    json.dump(history.history, outfile)
    outfile.close()

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO] serializing model...")
