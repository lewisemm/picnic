import os
import shutil
import utils

# categorize the photos into folders. Each folder represents a category
filename = 'picnic_data/train.tsv'
dir_prefix = 'picnic_data/train/'

labels = utils.unique_category_labels(filename)
len_labels = len(labels)
label_to_folder_map = {}

print("Creating train sub-directories...")
for ind, label in zip(range(len_labels), labels):
    os.mkdir(dir_prefix + str(ind))
    label_to_folder_map[label] = ind

opened = open(filename, 'r')
# skip the headers in the first line
opened.readline()

print("Moving photo files to appropriate subdirectories...")
for line in opened:
    name, label = line.split('\t')
    name = name.lower()
    label = label.lower()[:-1]

    dest = dir_prefix + str(label_to_folder_map[label])
    name = dir_prefix + name
        
    if label in label_to_folder_map:
        shutil.move(name, dest)