'''
Script to show audio statistics of an Audacity labeled audio dataset with multiple folders inside
@author Joan Gomez
'''
import os
import sys
import csv

EXTENDED_VERBOSE = 0

#if len(sys.argv) < 2:
#	print('Please pass the folder path')
#	exit()

if len(sys.argv) < 2:
	original_path = "./"
else:
	original_path = sys.argv[1]

if original_path[len(original_path) - 1] != '/':
	original_path += "/"

print("\n-------------------------------- Audio dataset statistics -------------------------------")
for dir in os.listdir(original_path):
	folder_path = original_path + dir

	# If the next element is not a dir, pass
	if not os.path.isdir(folder_path):
		continue

	# Loop all files in a directory and compute all .txt
	totalSeconds = 0
	totalCuts = 0
	if EXTENDED_VERBOSE:
		print("\nDirectory:", folder_path)
	for i in os.listdir(folder_path):

		if i.endswith('.txt'):
			fileSeconds = 0
			fd = open(folder_path + '/' + i)
			reader = csv.reader(fd, delimiter='\t')

			# Add all durations from a txt file
			for row in reader:
				diff = float(row[1]) - float(row[0])
				fileSeconds += diff
				totalCuts += 1
			fd.close()
			totalSeconds += fileSeconds
			if EXTENDED_VERBOSE:
				print(i, "has", round(fileSeconds, 2), "seconds of audio.")

	if totalCuts == 0:
		print(folder_path[2:].ljust(25), "| Duration (min):", str(round(totalSeconds/60, 2)).zfill(5), "| Cuts:", str(totalCuts).zfill(3), 
			"| Duration (avg sec): 00.00")
	else:
		print(folder_path[2:].ljust(25), "| Duration (min):", str(round(totalSeconds/60, 2)).zfill(5), "| Cuts:", str(totalCuts).zfill(3), 
			"| Duration (avg sec):", str(round(totalSeconds/totalCuts,2)).zfill(5))
