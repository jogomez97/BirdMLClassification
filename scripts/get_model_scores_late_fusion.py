import os
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from imutils import paths
from sklearn.model_selection import train_test_split

from custom_dataset_loader_and_splitter import CustomDatasetLoaderAndSplitter
from simpledatasetloader import SimpleDatasetLoader
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor

from custom_fusion_dataset_loader_and_splitter import CustomFusionDatasetLoaderAndSplitter

args = {
    "dataset_audio": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s',
    "dataset_image": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Imatge/02_Base de dades/training', 
    "dataset_image_val": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Imatge/02_Base de dades/validation',
    "model_a": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez\Models/transfer_learning_vgg16_split_test/output.h5',
    "model_i": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez\Models/transfer_learning_image_vgg16/weights-improvement-38-0.85.h5'
}

batch_size = 16

# load both models
model_a = load_model(args['model_a'])
model_i = load_model(args['model_i'])

# get labels
classNames = list(os.listdir(args["dataset_image"]))

# get audio + image data
c = CustomFusionDatasetLoaderAndSplitter(args["dataset_audio"], 
    image_path=args["dataset_image"], 
    image_path_val=args["dataset_image_val"],
    validation=0.2, included_folders=classNames, only_val=True)

testX, testY = c.load_and_split()

# convert the labels from integers to vectors
testY = LabelBinarizer().fit_transform(testY)

audios = []
images = []
for t in testX:
    # audio is on top of the image
    audios.append(t[0:224])
    images.append(t[224:])

audios = np.array(audios)
images = np.array(images)

# evaluate both networks
print("[INFO] evaluating audio model")
predictions_a = model_a.predict(audios, batch_size=batch_size)
print("[INFO] evaluating image model")
predictions_i = model_i.predict(images, batch_size=batch_size)

# audio model has 3 more classes, we need to delete them
predictions_a = np.delete(predictions_a, [1, 4, 7], axis=1)

# add predictions from both branches to later find the arg maximum
predictions = predictions_a + predictions_i

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
c_dict = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames, output_dict=True)
avg_score = c_dict['macro avg']['recall']

ax = skplt.metrics.plot_confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1), normalize=True, cmap='Blues', figsize=(12,8))
ax.xaxis.set_ticklabels(classNames); ax.yaxis.set_ticklabels(classNames)
plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
plt.tight_layout()
ax.set_xlabel('Predicted label\nAverage score: {:.03f}'.format(avg_score))
plt.show()