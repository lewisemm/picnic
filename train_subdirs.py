import json
import os
import shutil

import utils

# Categorize the photos into sub-folders.
# Each sub-folder represents a category.
filename = 'picnic_data/train.tsv'
dir_prefix = 'picnic_data/train/'

labels = utils.unique_category_labels(filename)
label_to_folder_map = {}

print("Creating train sub-directories...")
for ind, label in enumerate(labels):
    os.mkdir(dir_prefix + str(ind))
    label_to_folder_map[label] = ind

opened = open(filename, 'r')
# skip the headers in the first line
opened.readline()

print("Moving photo files to sub-folders based on label...")
for line in opened:
    photo_filename, label = line.split('\t')
    photo_filename = photo_filename.lower()
    label = label.lower()[:-1]

    dest = dir_prefix + str(label_to_folder_map[label])
    photo_filename = dir_prefix + photo_filename
        
    if label in label_to_folder_map:
        shutil.move(photo_filename, dest)

# persist the label to folder mapping
with open('label_to_folder_map.json', 'w') as file:
    file.write(json.dumps(label_to_folder_map)) 