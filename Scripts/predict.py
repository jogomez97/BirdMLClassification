from keras.models import load_model
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the file to be tested")
ap.add_argument("-l", "--labels", required=True, help="path to the labels file")
ap.add_argument("-m", "--model", required=True, help="path to the model")
args = ap.parse_args()

# process the input numpy array
image = np.load(args.input) / 255.0
image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
image = np.expand_dims(image, axis=0)

# load all the possible labels from file
labels = open(args.labels).read().splitlines()

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(args.model)

# pass the image through the network to obtain our predictions
preds = model.predict(image)[0]
i = np.argmax(preds)
label = labels[i]
print(preds)
print(label)