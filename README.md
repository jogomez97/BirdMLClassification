# BirdMLClassification
The aim of this work is to study the ability to use machine learning algorithms generally applied to image recognition for the classification of sounds of 20 different species of birds in the Aiguamolls de l'Empordà natural park in Catalonia, Spain.

For this purpose, it was decided to use different pre-trained CNNs and a set of manually labelled audios, processed in the form of spectrograms, to obtain different models that are able to predict with a high degree of confidence which species of bird is emitting sounds.

Additionally, a model to recognize bird species through images is accomplished. With both audio and image based models, several methods are studied to increase the accuracy of the classification by combining (or fusing) image and sound.

## Dataset
The audio dataset is published and accessible at: [Western Mediterranean Wetlands Bird Dataset](https://zenodo.org/record/5093173).

## Folders
- [Scripts](scripts): has the main scripts used for the project. This includes transforming audio dataset to spectrograms, loss and accuracy evolution plotters or confusion matrix plotters among others.
- [Transfer learning](transfer_learning): has all the scripts concerning training networks via fine-tuning (including early fusion).
- [Models](models): all trained models from the project. 
    - [Audio models](models/audio): models trained using a spectrogram dataset using fine-tuning. 
    VGG16 image fine-tuning. VGG16 early fusion.
    - [Image models](models/image): models trained with bird photos using fine-tuning.
    - [Fusion models](models/fusion): models trained with both photos and spectrograms.

## List of bird species
- Acrocephalus arundinaceus
- Acrocephalus melanopogon
- Acrocephalus scirpaceus
- Alcedo atthis
- Anas strepera
- Anas platyrhynchos
- Ardea purpurea
- Botaurus stellaris
- Charadrius alexandrinus
- Ciconia ciconia
- Circus aeruginosus
- Coracias garrulus
- Dendrocopos minor
- Fulica atra
- Gallinula chloropus
- Himantopus himantopus
- Ixobrychus minutus
- Motacilla flava
- Porphyrio porphyrio
- Tachybaptus ruficollis