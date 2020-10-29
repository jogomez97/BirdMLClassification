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

# AUDIO
# MODEL_PATH = 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez\Models/transfer_learning_vgg16_split_test/output.h5'
# MODEL_PATH = 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez\Models/transfer_learning_resnet50_split_test/weights-improvement-20-0.86.h5'
# MODEL_PATH = 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez\Models/transfer_learning_mobilenetv2_split_test/weights-improvement-51-0.83.h5'
# DATASET_PATH = 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s'

# IMAGE
MODEL_PATH = 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Models/transfer_learning_image_vgg16/weights-improvement-38-0.85.h5'
DATASET_PATH = '../Imatge/02_Base de dades/full/'

is_audio = False

batch_size = 16

model = load_model(MODEL_PATH)

# get the test images and labels
classNames = list(os.listdir(DATASET_PATH))

if is_audio:
    c = CustomDatasetLoaderAndSplitter(DATASET_PATH, validation=0.2, test=0.1)
    testX, testY = c.load_and_split(only_test=True)
else:
    imagePaths = list(paths.list_images(DATASET_PATH))

    aap = AspectAwarePreprocessor(224, 224)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (data, labels) = sdl.load(imagePaths, verbose=200)
    data = data.astype("float") / 255.0
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

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
