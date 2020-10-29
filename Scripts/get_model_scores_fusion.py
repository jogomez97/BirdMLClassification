'''
early fusion scores
'''
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

from custom_fusion_dataset_loader_and_splitter import CustomFusionDatasetLoaderAndSplitter

args = {
    "dataset_audio": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s',
    "dataset_image": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Imatge/02_Base de dades/training', 
    "dataset_image_val": '../Imatge/02_Base de dades/full/',
    "model_path": 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Models/transfer_learning_fusion_vgg16/weights-improvement-35-0.75.h5'
}

batch_size = 16

model = load_model(args["model_path"])

# get the test images and labels
classNames = list(os.listdir(args["dataset_image"]))

c = CustomFusionDatasetLoaderAndSplitter(args["dataset_audio"], 
    image_path=args["dataset_image"], 
    image_path_val=args["dataset_image_val"],
    validation=0.2, included_folders=classNames, only_val=True)
testX, testY = c.load_and_split()

# convert the labels from integers to vectors
testY = LabelBinarizer().fit_transform(testY)

lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
c_dict = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames, output_dict=True)
avg_score = c_dict['macro avg']['recall']

ax = skplt.metrics.plot_confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1), normalize=True, cmap='Blues', figsize=(12,8))
ax.xaxis.set_ticklabels(classNames); ax.yaxis.set_ticklabels(classNames)
plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
plt.tight_layout()
ax.set_xlabel('Predicted label\nAverage score: {:.03f}'.format(avg_score))
plt.show()
