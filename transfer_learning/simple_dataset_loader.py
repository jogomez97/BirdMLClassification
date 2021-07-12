# Imports
import numpy as np
import cv2
import os
from PIL import Image
import random

class SimpleDatasetLoader:

    def __init__(self, size=224):
        self.size = size

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        i = 0
        # loop over the input images
        for dir in os.listdir(imagePaths):
            folder_path = os.path.join(imagePaths, dir)
            for img_name in os.listdir(folder_path):
                imagePath = os.path.join(folder_path, img_name)

                # load the image and extract the class label assuming that our path has the following format: /path/to/dataset/{class}/{image}.jpg
                try:
                    image = Image.open(imagePath)
                except Exception as e:
                    print(e)
                    continue
                image = image.convert('RGB')
                image = image.resize((self.size, self.size), Image.LANCZOS)
                image = np.array(image)
                image = image.astype("float32") / 255.0
                label = imagePath.split(os.path.sep)[-2]
                
                data.append(image)
                labels.append(label)
                i += 1

                # show an update every image processed
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {} {}\t\t".format(i + 1, label))
        
        # randomly shuffle the data
        aux_list = list(zip(data, labels))
        random.shuffle(aux_list)
        data, labels = zip(*aux_list)
        labels = np.array(labels)

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))