import argparse
import datetime

import utils


# don't want to keep typing this command on terminal
# python predict.py picnic_data/test/7263.jpeg picnic_data/train.tsv --arch alexnet --top_k 3 --gpu 1

parser = argparse.ArgumentParser()

default_image = "picnic_data/test/7263.jpeg"
cat_labels_path = "picnic_data/train.tsv"

parser.add_argument('image_path', type=str, default=default_image, help='the path to the image to be identified')
parser.add_argument('--arch', type=str, default='vgg16', help='the CNN Model Architecture')
parser.add_argument('--top_k', type=int, default=5, help='return the top k likely food categories')
parser.add_argument('--gpu', type=int, default=0, help="Use a GPU to predict")
parser.add_argument('--checkpoint_path', type=str, default='food_vgg16.pth',
    help='Save a checkpoint to a file with the specified name.')


args = parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # get the values from the params
    image_path = args.image_path
    arch = args.arch
    top_k = args.top_k
    gpu = args.gpu
    checkpoint_path = args.checkpoint_path

    model = utils.load_checkpoint(arch, checkpoint_path)
    model.eval()
    
    probs, labels, foods = utils.predict(image_path, model, top_k, gpu)
    
    stop_time = datetime.datetime.now()
    
    diff = stop_time - start_time
    print("*" * 80)
    for label, prob in zip(labels, probs):
        print("{}: {} %".format(label, round(prob * 100, 3)))
    print("*" * 80)
    print("Running time: {} seconds".format(diff.total_seconds()))
print("*" * 80)