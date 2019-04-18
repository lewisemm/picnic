import os
import shutil
import utils

from PIL import Image


test_file = "picnic_data/test.tsv"
dir_prefix = "picnic_data/test/"
arch = "vgg16"
checkpoint_path = "vgg16.pth"

opened = open(test_file, "r")
opened.readline()

sub = open("submission.tsv", "w")
sub.write("file\tlabel\n")


model = utils.load_checkpoint(arch, checkpoint_path)
model.eval()

for line in opened:
    testfile_name = line[:-1]

    top_probs, top_idx_names, top_categs = None, None, None

    if '.png' in testfile_name:
        # convert to jpg
        im = Image.open(dir_prefix + testfile_name)
        rgb_im = im.convert('RGB')

        testfile_name_jpg = dir_prefix + testfile_name[:-4] + ".jpg"
        
        rgb_im.save(testfile_name_jpg)
        top_probs, top_idx_names, top_categs = utils.predict(testfile_name_jpg, model, 1, 1)
        os.remove(testfile_name_jpg)
    else:
        top_probs, top_idx_names, top_categs = utils.predict(dir_prefix + testfile_name, model, 1, 1)

    completed_line = "{}\t{}\n".format(testfile_name, top_idx_names[0])
    try:
        sub.write(completed_line)
    except Exception:
        sub.close()
sub.close()