'''
Script that given an audio dataset it produces an equivalent one with it's spectograms
@author Joan Gomez
'''

# imports
import os
import argparse
import math
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# functions
def create_dir_if_not_exists(path):
    if not os.path.isdir(path):
        print("%s directory does not exist. Creating..." % path)
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
            raise


def process_librosa(input_path, output_path, offset, duration, n_mels, hop_length, arr_length):
    # librosa sets sample rate to 22.05kHz and normalizes the bit depth between -1 and 1
    # also it converts stereo to mono
    y, sr = librosa.load(input_path, offset=offset, duration=duration)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    spec = librosa.power_to_db(spec, ref=np.max)

    # print(spec.shape)
    # If duration < time_split then padding needs to be added
    # The begging of the audio will be replicated to the end until time_split is matched
    if spec.shape[1] < arr_length:
        spec = np.pad(spec, ((0, 0), (0, arr_length - spec.shape[1] % arr_length)), mode='wrap')
    if spec.shape[1] > arr_length:
        spec = np.delete(spec, np.s_[arr_length - spec.shape[1]:], axis=1)
    np.save(output_path, spec)
    
    # save_cut_audio(spec, sr, output_path + '.wav')
    plot_spectrogram(spec, sr)

def plot_spectrogram(spec, sr):
    librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar()
    plt.show()

def save_cut_audio(db_spec, sr, output_path):
    spec = librosa.db_to_power(db_spec)
    audio = librosa.feature.inverse.mel_to_audio(spec)
    librosa.output.write_wav(output_path, audio, sr)

def check_percentage(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 0.5:
        raise argparse.ArgumentTypeError("%s is an percentage in range [0 0.5]" % value)
    return fvalue
########
# MAIN #
########

print("------ Batch Spectogram Converter ------")
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the dataset directory")
ap.add_argument("-o", "--output", required=False,help="path to the output directory. Default ./img_dataset", default='./img_dataset')
ap.add_argument("-s", "--split", required=False, help="split time interval in seconds. Default 1 sec", default=1.0, type=float)
ap.add_argument("-ov", "--overlap", required=False, help="overlap percentage in case of multiple splits. Must be between 0 and 0.5. Default 0.2", default=0.2, type=check_percentage)
ap.add_argument("-m", "--mels", required=False, help="number of mel filers. Default 224", default=224)
args = vars(ap.parse_args())

n_mels = int(args["mels"])
time_split = float(args["split"])

# dimensions for the output array must be square to fit as inputs for CNNs
# n_mels = shape[0] of mel spectogram
# time_split * sample_rate / hop_length  = shape[1] of mel spectogram
# calculate arr_length based on https://stackoverflow.com/questions/62584184/trying-to-understand-shape-of-spectrograms-and-n-mels
hop_length = time_split * 22050 / n_mels
arr_length = math.ceil(time_split * 22050 / hop_length)
hop_length = math.ceil(hop_length)

# check if input folder exists
if not os.path.isdir(args["input"]):
    print("Input directory does not exist")
    exit(-1)

# check if output folder exists. If not create it
create_dir_if_not_exists(args["output"])

f = open("stats.txt", "w")
# loop through all dataset directories
for dir in os.listdir(args["input"]):
    audio_folder = os.path.join(args["input"], dir)
    spectrogram_folder = os.path.join(args["output"], dir)
    n_audio_folder = 0

    if dir == '.' or dir == '..' or not os.path.isdir(audio_folder):
        continue

    # check if output sub-folder exists. If not create it
    create_dir_if_not_exists(spectrogram_folder)

    print("\033[94m* Processing %s folder\033[0m" % audio_folder)
    # Loop all files in the directory. Find the .txt and create the spectogram
    for file_name in os.listdir(audio_folder):
        if file_name.endswith('.txt'):
            labels_file = os.path.join(audio_folder, file_name)
            audio_file = labels_file.split('.txt')[0] + '.mp3'
            spectrogram_file = os.path.join(spectrogram_folder, file_name.split(".txt")[0])

            print("\t\033[93m* Processing %s file\033[0m" % labels_file)
            # process every line of the labels txt file
            with open(labels_file) as fd:
                lines = fd.readlines()
                n_audio = -1
                for line in lines:
                    line_split = line.split("\t")
                    t1 = float(line_split[0])
                    t2 = float(line_split[1])
                    t_diff = t2 - t1
                    # the audio cut length is less than the specified time split
                    if t_diff <= time_split:
                        n_audio += 1
                        new_file = spectrogram_file + '_' + str(n_audio)
                        print("\t\t* one split of %f sec" % t_diff)
                        process_librosa(audio_file, new_file, t1, t_diff, n_mels, hop_length, arr_length)
                    # the audio cut fits multiple time splits (ex: 2.5sec audio in a 1sec time split)
                    elif t_diff > time_split:
                        # in this case, we don't perfectly cut every time split. Instead, we go up every time split - X
                        t_arr = np.arange(start=t1, stop=t2, step=time_split * (1 - args["overlap"]))
                        print("\t\t* multiple splits:", t_arr)
                        for i, t in enumerate(t_arr):
                            if i == len(t_arr) - 1:
                                if t2 - t > 0.2:
                                    n_audio += 1
                                    new_file = spectrogram_file + '_' + str(n_audio)
                                    process_librosa(audio_file, new_file, t, t2 - t, n_mels, hop_length, arr_length)
                            else:
                                n_audio += 1
                                new_file = spectrogram_file + '_' + str(n_audio)
                                process_librosa(audio_file, new_file, t, time_split, n_mels, hop_length, arr_length)
                n_audio_folder += n_audio + 1
                print("\t\033[92m* Created %d cuts of audio. Accumulative: %d\033[0m" % ((n_audio + 1), n_audio_folder))
    f.write(spectrogram_folder)
    f.write(" - ")
    f.write(str(n_audio_folder))
    f.write("\n")
                        
f.close()
print("Done")                    