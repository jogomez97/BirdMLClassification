'''
Script to rename all auto-downloaded files from xenocanto with name: XCNNNNN - Common name - latin name.mp3
to just XCNNNNN.mp3
@author: Joan Gomez
'''

import os
import sys

if len(sys.argv) < 2:
	original_path = "C:/Users/VersusPC/Downloads/downthemall" 
else:
	original_path = sys.argv[1]	

if original_path[len(original_path) - 1] != '/':
	original_path += "/"

for filename in os.listdir(original_path):
	list = filename.split(" - ")
	if len(list) == 3:
		os.rename(original_path + filename, original_path + list[0] + ".mp3")
