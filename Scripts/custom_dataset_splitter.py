# imports
import os
import argparse
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the dataset directory")
ap.add_argument("-o", "--output", required=False, help="path to the output file. Default ./stats.txt", default='./stats.txt')
ap.add_argument("-v", "--validation", required=False, type=float, help="split amount for valiation. Default 0.25", default=0.25)
ap.add_argument("-t", "--test", required=False, type=float, help="split amount for test. Default 0", default=0)
args = ap.parse_args()

train_split = round(1 - (args.validation + args.test), 2)
print("Input split: train {}%, validation {}%, test {}%".format(train_split * 100, args.validation * 100, args.test * 100))
print("===== Dataset =====")
fd = open(args.output, 'w')
cuts_per_file = {}
for dir in os.listdir(args.input):
    folder_path = os.path.join(args.input, dir)

    # If the next element is not a dir, pass
    if not os.path.isdir(folder_path):
        continue

    # Loop through all files and count number of cuts from the same audio
    cuts_per_file[dir] = {}
    for file_name in os.listdir(folder_path):
        audio_name = file_name.split('_')[0]
        if audio_name in cuts_per_file[dir]:
            cuts_per_file[dir][audio_name] += 1
        else:
            cuts_per_file[dir][audio_name] = 1
    
# Write statistics of directory in the file
fd.write(json.dumps(cuts_per_file))
fd.write('\n')
fd.close()

train_dict = {}
val_dict = {}
test_dict = {}
real_train_split = train_split
real_val_split = args.validation
real_test_split = args.test
for d, folder in cuts_per_file.items():
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

        if p_total < train_split:
            train_dict[d].append(x)
            prob_dict.pop(x)
            continue
        if p_total == train_split or (p_total > train_split and p_total <= train_split + 0.01):
            train_dict[d].append(x)
            prob_dict.pop(x)
            break
        else:
            p_total -= p

    real_train_split = p_total * 100
    
    # If no test needs to be added, the remaining data is validation itself
    if args.test == 0:
        val_dict[d] = list(prob_dict.keys())
        real_val_split = round(100 - real_train_split, 3)
        print("Real split: train {}%, validation {}%, test {}%".format(real_train_split, real_val_split, 0))
        continue
    
    # Get validation data
    p_total = 0
    val_dict[d] = []
    for x, p in list(prob_dict.items()):
        p_total += p
        p_total = round(p_total, 2)

        if p_total < args.validation:
            val_dict[d].append(x)
            prob_dict.pop(x)
            continue
        if p_total == args.validation or (p_total > args.validation and p_total <= args.validation + 0.01):
            val_dict[d].append(x)
            prob_dict.pop(x)
            break
        else:
            p_total -= p

    real_val_split = round(p_total * 100, 3)

    # The rest of the files will be for test
    test_dict[d] = list(prob_dict.keys())
    real_test_split = round(100 - (real_train_split + real_val_split), 3)
    print("Real split: train {}%, validation {}%, test {}%".format(real_train_split, real_val_split, real_test_split))

print(test_dict)
    

