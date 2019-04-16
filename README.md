# Picnic - When great customer support meets data

The challenge this app aims to solve is the following: 

* As a first step in helping customer support, come up with a way of labeling every picture that comes in according to the product that is in the picture. To keep with the Picnic spirit, we encourage to be as innovative and creative with your solutions as possible.

More details can be found [here](https://picnic.devpost.com/).

## Workspace Setup
The steps below outline how to setup the workspace for image training and/or image classification.

1. Clone this repository.

    ```bash
    git clone git@github.com:lewisemm/picnic.git
    ```
2. `cd` into the `picnic` repository.

    ```bash
    cd picnic
    ```
3. Download the zipped picnic hackathon dataset [here](https://drive.google.com/file/d/1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC/view?usp=sharing). The file size is approximately 3GB.
4. Once the download is complete, copy the zipped file into the `picnic` repository.

    ```bash
    cp ~/Downloads/The Picnic Hackathon 2019.zip ~/projects/picnic/
    ```
5. Unzip the compressed `The Picnic Hackathon 2019.zip` file.

    ```bash
    unzip The\ Picnic\ Hackathon\ 2019.zip
    ```
6. Rename the unzipped directory from `The Picnic Hackathon 2019` to `picnic_data`. This is done to shorten the directory name and eliminate the spaces in its name.

    ```bash
    mv The\ Picnic\ Hackathon\ 2019 picnic_data
    ```
7. [Select, download and install Anaconda](https://www.anaconda.com/distribution/) for your platform. Installation instructions can be found [here](https://docs.anaconda.com/anaconda/install/).
8. Use `conda` to create the `picnic` environment.

    ```bash
    conda env create -f environment.yml
    ```
9. Run the following command to split the photos in `picnic_data/train` into category specific subdirectories.

    ```bash
    python train_subdirs.py
    ```

## Training the model
[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is used to train the image classifier using pre-trained models. Three pre-trained models are available for training;
* alexnet
* vgg13
* vgg16

Training parameters can be supplied from the command line when the default values are not desired. Available training options are listed below.
* **data_dir** - This is the path to the directory containing the dataset. Defaults to `picnic_data`. (Required)
* **--checkpoint_path** - Save a checkpoint to a file with the name specified here. Defaults to `checkpoint.pth`.
* **--arch** - Selects the pre-trained model architecture to be used in training. Defaults to `vgg16`.
* **--learning_rate** - Sets the learning rate to be used when the network is training. Defaults to `0.01`.
* **--dropout** - Sets the dropout rate to be used when training the network. Defaults to `0.3`.
* **--hidden_units** - Sets the hidden layers to be used when training the model. For more than one layer, each layer's value will be space separated. Defaults to `10000`.
* **--epochs** - The number of epochs to use when training the network. Defaults to `7`.
* **--gpu** - When set to 0, a GPU will not be used in the model training process. Defaults to `1`.

### Example model training usage

```bash
python train.py picnic_data --checkpoint_path alexnet.pth --arch alexnet --learning_rate 0.001 --dropout 0.40 --hidden_units 10000 5000 2500 --epochs 5 --gpu 1
```

## Making Predictions
> Training has to take place before predictions can be made. Predictions cannot work without the checkpoint file ('*.pth') that is generated from training the model.

Predictions, like training, is also done via the command line. Available options for the prediction function include the following;
* **image_path** - Specifies the path to the image which needs to be classified. Defaults to `picnic_data/test/7263.jpeg`.
* **--arch** - Specifies the CNN model architecture that was used to train the checkpoint file. Defaults to `vgg16`.
* **--top_k** - return the top k likely predictions. Defaults to `5`.
* **--gpu** - Opt to use a GPU when making a prediction. Defaults to `0`.
* **--checkpoint_path** - Specifies the model checkpoint file to use. Defaults to `vgg16.pth`.

### Example prediction usage

```bash
python predict.py picnic_data/test/7263.jpeg --arch vgg16 --top_k 3 --gpu 1 --checkpoint_path vgg16.pth

********************************************************************************
bell peppers, zucchinis & eggplants: 75.478 %
bananas, apples & pears: 17.594 %
cucumber, tomatoes & avocados: 4.688 %
********************************************************************************
Running time: 8.06721 seconds
********************************************************************************
```

## Hackathon Submission
The submission process requires that the filenames listed in `test.tsv` be predicted and the results stored in `.tsv` format.

Running the following command will take care of the above requirement.

```bash
python submitter.py
```

This generates a `submission.tsv` file that contains the filenames and the corresponding classification of images in those files.