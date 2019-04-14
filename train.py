import argparse

from torch import nn

import utils


# don't want to keep typing this command on terminal
# python train.py picnic_data --checkpoint_path food_vgg16.pth --arch vgg16 --learning_rate 0.001 --dropout 0.40 --epochs 5 --hidden_units 10000 5000 2500 --gpu 1

parser = argparse.ArgumentParser()
    
parser.add_argument('data_dir', type=str, default = 'picnic_data',
    help='the directory which contains training, validation and images')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth',
    help='Save a checkpoint to a file with the specified name.')
parser.add_argument('--arch', type=str, default='vgg16', help='the Model Architecture')
parser.add_argument('--learning_rate', type=float, default=0.01,
    help='the learning rate while training the network')
parser.add_argument('--dropout', type=float, default=0.3,
    help='the dropout rate while training the network')
parser.add_argument('--hidden_units', type=int, nargs='+', default=10000,
    help="space separated integer values for each hidden unit", required=True)
parser.add_argument('--epochs', type=int, default=7,
    help="the number of epochs used while training the network")
parser.add_argument('--gpu', type=int, default=1, help="Use a GPU to train the network")

args = parser.parse_args()

if __name__ == "__main__":
    # get the values from the params
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    arch = args.arch
    lr = args.learning_rate
    dropout = args.dropout
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    print("These are the arguments supplied! :\n {}".format(args))
    
    image_datasets, dataloaders = utils.create_dataloaders(data_dir)
    
    # nn model stuff
    supported_archs = utils.get_supported_archs()
    
    if arch not in supported_archs:
        print("'{}' not supported. Please choose either vgg16 or alexnet".format(arch))
    else:
        pretrained_model = supported_archs[arch][0]
        model_input = supported_archs[arch][1]

        food_classifier = utils.setup_nn(pretrained_model, model_input, hidden_units, dropout)
        
        # train and save the model to file
        trained_food_classifier = utils.train_model(food_classifier, gpu, dataloaders, lr=lr, epochs=epochs)
        utils.save_checkpoint(trained_food_classifier, image_datasets, hidden_units, checkpoint_path)
