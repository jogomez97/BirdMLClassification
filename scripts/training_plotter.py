import matplotlib.pyplot as plt
import json 
import argparse
from distutils.util import strtobool

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the training file")
ap.add_argument("-n", "--name", required=False, help="title of the plot", default="")
ap.add_argument("-e", "--extra", required=False, help="path to secondary warmup file")
ap.add_argument("-l", "--loss", required=False, type=lambda x:bool(strtobool(x)),  nargs='?', const=True, help="include loss. Default False", default=False)
ap.add_argument("-a", "--accuracy", required=False, type=lambda x:bool(strtobool(x)),  nargs='?', const=True, help="include accuracy. Default True", default=True)
ap.add_argument("-v", "--vertical", required=False, type=lambda x:bool(strtobool(x)),  nargs='?', const=True, help="include vertical line after warmup", default=False)
args = ap.parse_args()

train_loss = []
train_acc = []
val_loss = []
val_acc = []
warmup_epochs = 0

if args.loss == False and args.accuracy == False:
    raise ValueError('Please set loss or accuracy arguments to True. Type -h for help.')

if args.extra is not None:
    with open(args.extra) as json_file:
        data = json.load(json_file)
        try:
            train_acc.extend(data['accuracy'])
        except:
            train_acc.extend(data['acc'])
        train_loss.extend(data['loss'])
        val_loss.extend(data['val_loss'])
        try:
            val_acc.extend(data['val_accuracy'])
        except:
            val_acc.extend(data['val_acc'])
        warmup_epochs = len(train_acc)

# read files
with open(args.input) as json_file:
    data = json.load(json_file)
    try:
        train_acc.extend(data['accuracy'])
    except:
        train_acc.extend(data['acc'])
    train_loss.extend(data['loss'])
    val_loss.extend(data['val_loss'])
    try:
        val_acc.extend(data['val_accuracy'])
    except:
            val_acc.extend(data['val_acc'])

# Create count of the number of epochs
epoch_count = range(1, len(train_loss) + 1)

# Visualize loss history
if args.accuracy == True and args.loss == True:
    plt.plot(epoch_count, train_acc, 'b-')
    plt.plot(epoch_count, train_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'g--')
    plt.plot(epoch_count, val_acc, 'y-')
    plt.legend(['Training accuracy', 'Training loss', 'Validation loss', 'Validation accuracy'])
    plt.ylabel('Loss/Accuracy')
elif args.accuracy == True and args.loss == False:
    plt.plot(epoch_count, train_acc, 'b-')
    plt.plot(epoch_count, val_acc, 'y-')
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.ylabel('Accuracy')
else:
    plt.plot(epoch_count, train_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'g--')
    plt.legend(['Training loss', 'Validation loss'])
    plt.ylabel('Loss')    

if args.vertical == True and warmup_epochs > 0:
    plt.axvline(warmup_epochs)

plt.xlabel('Epoch')
plt.title(args.name)
plt.show()