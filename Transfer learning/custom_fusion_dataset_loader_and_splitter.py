# imports
import os
import argparse
import json
import numpy as np
import random
from PIL import Image

class CustomFusionDatasetLoaderAndSplitter:

    def __init__(self, input_path, image_path, image_path_val, validation=0.25, test=0, verbose=False, included_folders=[], image_size=224):
        self.input = input_path 
        self.image_path = image_path
        self.image_path_val = image_path_val
        self.validation = validation
        self.test = test 
        self.verbose = verbose
        self.included_folders = included_folders
        self.image_size = image_size

        if self.validation < 0 or self.validation > 1:
            raise ValueError('Error, validation must be a float between 0 and 1')
        if self.test < 0 or self.test > 1:
            raise ValueError('Error, test must be a float between 0 and 1')

        self.train_split = round(1 - (self.validation + self.test), 2)
        if self.train_split < 0:
            raise ValueError('Error, validation and test can\'t add to more than 1')

        print("Input split: train {}%, validation {}%, test {}%".format(self.train_split * 100, self.validation * 100, self.test * 100))
        if self.verbose:
            print("===== Dataset =====")

    def __split(self):
        cuts_per_file = {}
        for dir in os.listdir(self.input):
            folder_path = os.path.join(self.input, dir)

            # If the next element is not a dir, pass
            if not os.path.isdir(folder_path):
                continue

            # If we only want certain classes to be loaded
            if len(self.included_folders) > 0:
                if dir not in self.included_folders:
                    continue

            # Loop through all files and count number of cuts from the same audio
            cuts_per_file[dir] = {}
            for file_name in os.listdir(folder_path):
                audio_name = file_name.split('_')[0]
                if audio_name in cuts_per_file[dir]:
                    cuts_per_file[dir][audio_name] += 1
                else:
                    cuts_per_file[dir][audio_name] = 1
            
        train_dict = {}
        val_dict = {}
        test_dict = {}
        real_train_split = self.train_split
        real_val_split = self.validation
        real_test_split = self.test
        for d, folder in cuts_per_file.items():
            if self.verbose:
                print(d)
            # calculate the sum of cuts
            count = 0
            for n in folder.values():
                count += n
            
            # calculate percentages of audio files
            prob_dict = {}
            for e, n in folder.items():
                prob_dict[e] = round(n / count, 4)

            # Get train data
            p_total = 0
            train_dict[d] = []
            for x, p in list(prob_dict.items()):
                p_total += p
                p_total = round(p_total, 2)

                if p_total < self.train_split:
                    train_dict[d].append(x)
                    prob_dict.pop(x)
                    continue
                if p_total == self.train_split or (p_total > self.train_split and p_total <= self.train_split + 0.01):
                    train_dict[d].append(x)
                    prob_dict.pop(x)
                    break
                else:
                    p_total -= p

            real_train_split = p_total * 100
            
            # If no test needs to be added, the remaining data is validation itself
            if self.test == 0:
                val_dict[d] = list(prob_dict.keys())
                real_val_split = round(100 - real_train_split, 3)
                if self.verbose:
                    print("Real split: train {}%, validation {}%, test {}%".format(real_train_split, real_val_split, 0))
                continue
            
            # Get validation data
            p_total = 0
            val_dict[d] = []
            for x, p in list(prob_dict.items()):
                p_total += p
                p_total = round(p_total, 2)

                if p_total < self.validation:
                    val_dict[d].append(x)
                    prob_dict.pop(x)
                    continue
                if p_total == self.validation or (p_total > self.validation and p_total <= self.validation + 0.01):
                    val_dict[d].append(x)
                    prob_dict.pop(x)
                    break
                else:
                    p_total -= p

            real_val_split = round(p_total * 100, 3)

            # The rest of the files will be for test
            test_dict[d] = list(prob_dict.keys())
            real_test_split = round(100 - (real_train_split + real_val_split), 3)
            if self.verbose:
                print("Real split: train {}%, validation {}%, test {}%".format(real_train_split, real_val_split, real_test_split))
        
        return train_dict, val_dict, test_dict
    
    def __get_all_images(self, old_dict):
        # update with real names of files
        updated_dict = {}
        
        # for label in dict get the images array
        for label, arr in old_dict.items(): 
            updated_dict[label] = []
            image_path = os.path.join(self.input, label)
            # for audio in folder
            for file_name in os.listdir(image_path):
                audio_name = file_name.split('_')[0]
                if audio_name in arr:
                    updated_dict[label].append(file_name)

        return updated_dict
        
    def __load(self, image_dict, is_val, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images stores as numpy arrays
        for label, arr in image_dict.items(): 
            i = 0
            label_path = os.path.join(self.input, label)
            for image_name in arr:
                image_path = os.path.join(label_path, image_name)
                # load the numpuy matrix and extract the class label assuming that our path has the following format: /path/to/dataset/{class}/{image}.npy
                image = np.load(image_path) / 255.0

                # input needs to be 3d, as our spectogram was a numpy matrix, we need to replicate it to form the 3rd dimension
                # example: (224, 224) image converts to (224, 224, 3) shape
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                
                # load random same class image from bird image dataset (not spectogram)
                birdImage = self.__load_random_image(is_val, label)

                # combine spectogram image and birdImage to have a big matrix of (image.shape + birdImage.shape, image.shape, 3)
                # limitation: numpy arrays must be the same size when stacked
                image = np.vstack((image, birdImage))

                data.append(image)
                labels.append(label)
                i += 1

                # show an update every image processed
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("\r[INFO] processed {}/{} {}\t\t\t".format(i + 1, len(arr), label), end='')

        print('')
        # randomly shuffle the data
        aux_list = list(zip(data, labels))
        random.shuffle(aux_list)
        data, labels = zip(*aux_list)
        labels = np.array(labels)

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

    def __load_random_image(self, is_val, label):
            if is_val:
                imagePath = os.path.join(self.image_path_val, label)
            else:
                imagePath = os.path.join(self.image_path, label)
           

            while True:
                # randomly pick an image of the folder
                imagePath = os.path.join(imagePath, random.choice(os.listdir(imagePath)))
                
                # try to load the image assuming that our path has the following format: /path/to/dataset/{class}/{image}.jpg
                try:
                    image = Image.open(imagePath)
                except Exception as e:
                    print(e)
                    continue

                image = image.convert('RGB')
                image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
                image = np.array(image)
                image = image.astype("float32") / 255.0
                break

            return image

    def load_and_split(self):
        # get the split of images
        print("Splitting...")
        self.train_dict, self.val_dict, self.test_dict = self.__split()

        # get the real names of images
        print("Getting all names for train...")
        self.train_dict = self.__get_all_images(self.train_dict)
        print("Getting all names for validation...")
        self.val_dict = self.__get_all_images(self.val_dict)
        print("Getting all names for test...")
        self.test_dict = self.__get_all_images(self.test_dict)
        
        # load the images in the three splits
        print("Loading train images...")
        X_train, Y_train = self.__load(self.train_dict, is_val=False,verbose=100)
        print("Loading validation images...")
        X_val, Y_val = self.__load(self.val_dict, is_val=True, verbose=50)
        if self.test > 0:
            print("Loading test images...")
            X_test, Y_test = self.__load(self.test_dict, is_val=True, verbose=20)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test

        return X_train, Y_train, X_val, Y_val


#############
# TEST AREA #
#############
'''      
f = open('./labels2.txt' , "r")
aux = f.readlines()
labels = []
for l in aux:
    labels.append(l.replace('\n', ''))
f.close()

c = CustomFusionDatasetLoaderAndSplitter('../Dataset espectogrames/birds_spec_224_224_1s/', 
    image_path='../Imatge/02_Base de dades/training', 
    image_path_val='../Imatge/02_Base de dades/training',
    validation=0.2, included_folders=labels)
c.load_and_split()
'''