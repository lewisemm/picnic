import argparse

from torch import nn

import utils


# don't want to keep typing this command on terminal
# python train.py picnic_data --checkpoint_path food_vgg16.pth --arch vgg16 --learning_rate 0.001 --dropout 0.40 --epochs 5 --hidden_units 10000 5000 2500 --gpu 1

parser = argparse.ArgumentParser()
    
parser.add_argument('data_dir', type=str, default = 'picnic_data',
    help='the directory which contains training, validation and images')
parser.add_argument('checkpoint_path', type=str, default='checkpoint.pth',
    help='the checkpoint path from which training will resume.')
parser.add_argument('--epochs', type=int, default=4,
    help="the number of additional epochs desired to further train a model")
parser.add_argument('arch', type=str, default='vgg16',
    help='the Model Architecture of the saved checkpoint')
parser.add_argument('--gpu', type=int, default=1, help="Use a GPU to train the network")

args = parser.parse_args()

if __name__ == "__main__":
    # get the values from the params
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    additional_epochs = args.epochs
    arch = args.arch
    gpu = args.gpu

    print("These are the arguments supplied! :\n {}".format(args))
    
    image_datasets, dataloaders = utils.create_dataloaders(data_dir)
    
    # nn model stuff
    supported_archs = utils.get_supported_archs()
    
    if arch not in supported_archs:
        print("'{}' not supported. Please choose either vgg16 or alexnet".format(arch))
    else:
        # resume training and save the model to file
        model = utils.resume_training(
            arch, checkpoint_path, additional_epochs, gpu, dataloaders)

        total_epochs = model.epochs + additional_epochs
        utils.save_checkpoint(
            model, image_datasets, model.hidden_units, checkpoint_path,
            model.dropout, total_epochs, models.learning_rate)
        