# Picnic - When great customer support meets data

The challenge this app aims to solve is the following: 

* As a first step in helping customer support, come up with a way of labeling every picture that comes in according to the product that is in the picture. To keep with the Picnic spirit, we encourage to be as innovative and creative with your solutions as possible.

More details can be found [here](https://picnic.devpost.com/).

## Workspace Setup
The steps below outline how to setup the workspace for image training and/or image classification.

1. Clone this repository.
    ```bash
    git@github.com:lewisemm/picnic.git
    ```
2. `cd` into the `picnic` repository.
    ```bash
    cd picnic
    ```
3. Download the picnic hackathon dataset [here](https://drive.google.com/file/d/1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC/view?usp=sharing). The file size is approximately 3GB.
4. Once the download is complete, move the zipped file into the `picnic` repository.
    ```bash
    mv ~/Downloads/The Picnic Hackathon 2019.zip ~/projects/picnic/
    ```
5. Unzip the compressed `The Picnic Hackathon 2019.zip` file.
    ```bash
    unzip The\ Picnic\ Hackathon\ 2019.zip
    ```
6. Rename the unzipped directory from `The Picnic Hackathon 2019` to `picnic_data`.
    ```bash
    mv The\ Picnic\ Hackathon\ 2019 picnic_data
    ```
7. [Select, download and install Anaconda](https://www.anaconda.com/distribution/) for your platform.
8. Use `conda` to create the `picnic` environment.
    ```bash
    conda env create -f environment.yml
    ```
9. Run the following command to split the photos in `picnic_data/train` into category specific subdirectories.
    ```bash
    python train_subdirs.py
    ```

## Training the model
[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is used to train the image classifier. This app has three options to use for transfer learning;
* alexnet
* vgg13
* vgg16

The training parameters can also be supplied from the command line. Available training options are listed below.
* **data_dir** - This is the path to the directory containing the dataset. Defaults to `picnic_data`. (Required)
* **--checkpoint_path** - Save a checkpoint to a file with the name specified here. Defaults to checkpoint.pth.
* **--arch** - Selects the pre-trained model architecture to be used in training. Defaults to vgg16.
* **--learning_rate** - Sets the learning rate to be used when the network is training. Defaults to 0.01.
* **--dropout** - Sets the dropout rate to be used when training the network. Defaults to 0.3.
* **--hidden_units** - Sets the hidden layers to be used when training the model. For more than one layer, each layer's value will be space separated.
* **--epochs** - The number of epochs to use when training the network. Defaults to 7.
* **--gpu** - When set to 0, a GPU will not be used in the model training process. Defaults to 1.

### Example model training usage

## Predictions
Predictions also allow for a fair degree of flexibility. Available options for the prediction function include the following;
*
*
*

### Example prediction usage