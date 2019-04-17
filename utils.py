import json
import numpy as np
import torch

from collections import OrderedDict
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models


def create_dataloaders(data_dir):
    train_and_valid_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data_transforms = {
        "train": train_and_valid_transforms,
        "valid": train_and_valid_transforms,
        "test": test_transforms,
    }
    
    # create the data loaders
    train_dir = data_dir + "/train/"
    valid_dir = data_dir + "/train/"
    test_dir = data_dir
    
    dirs = {
        "train": train_dir, 
        "valid": valid_dir, 
        "test" : test_dir
    }

    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in ["train", "valid", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ["train", "valid", "test"]}
    
    return image_datasets, dataloaders

def get_supported_archs():
    return {
        "vgg16": (models.vgg16(pretrained=True), 25088),
        # "vgg13": (models.vgg13(pretrained=True), 25088),
        # "alexnet": (models.alexnet(pretrained=True), 9216)
    }
        
def setup_nn(model, model_input, hidden_units, dropout):
    # maps the model string to the torchvision model and it's corresponding input
    # features at the classifier

    if len(hidden_units) != len(dropout):
        print("The number of hidden units should match the number of dropout rate values.")
    else:

        for param in model.parameters():
            param.requires_grad = False

        food_classifier = None

        if len(hidden_units) <= 2:
            food_classifier = nn.Sequential(
                OrderedDict([
                    ("fc1", nn.Linear(model_input, hidden_units[0])),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(p=dropout[0])),
                    ("fc2", nn.Linear(hidden_units[0], hidden_units[-1])),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(p=dropout[-1])),
                    ("fc3", nn.Linear(hidden_units[-1], 102)),
                    ("output", nn.LogSoftmax(dim=1))
                ])
            )
        elif len(hidden_units) > 2:
            fields = [
                ("fc1", nn.Linear(model_input, hidden_units[0])),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=dropout[0]))
            ]
            previous_input = hidden_units[0]
            dropout = dropout[1:]
            for index, unit in enumerate(hidden_units[1:]):
                fields.append(
                    ("fc{}".format(index + 2), nn.Linear(previous_input, unit))
                )
                fields.append(
                    ("relu{}".format(index + 2), nn.ReLU())
                )
                fields.append(
                    ("dropout{}".format(index + 2), nn.Dropout(p=dropout[index]))
                )
                previous_input = unit
            fields.append(
                ("fc{}".format(len(hidden_units) + 1), nn.Linear(hidden_units[-1], 26))
            )
            fields.append(
                ("output", nn.LogSoftmax(dim=1))
            )
            food_classifier = nn.Sequential(OrderedDict(fields))
        model.classifier = food_classifier
        return model

def determine_device(processor=1):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and processor == 1) else "cpu")
    return device

def validator(model, validation_loader, criterion, gpu):
    test_loss = 0
    accuracy = 0
                
    model.eval()
    device = determine_device(gpu)
    model.to(device)
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            probs = torch.exp(output)
            equality = (labels.data == probs.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    model.train()
    return test_loss, accuracy

def train_model(model, gpu, dataloaders, lr=0.01, epochs=7):
    device = determine_device(gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    print_every = 40
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for images, labels in dataloaders["train"]:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                loss, accuracy = validator(model, dataloaders['valid'], criterion, gpu)
                print(
                    "Epoch: {}/{}... ".format(e+1, epochs),
                    "Training Loss: {:.4f}".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(loss/len(dataloaders['valid'])),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid']))
                )

            print("Iteration {} of epoch {}".format(steps, e+1))

            running_loss = 0

    print("Training Complete!")
    return model

def resume_training(arch, checkpoint_filepath, additional_epochs, gpu, dataloaders):
    model = load_checkpoint(arch, checkpoint_filepath)

    device = determine_device(gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), model.learning_rate)

    print_every = 40
    steps = 0

    last_epoch = model.epochs + additional_epochs
    epoch_range = range(model.epochs, last_epoch)

    for e in epoch_range:
        running_loss = 0
        for images, labels in dataloaders["train"]:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                loss, accuracy = validator(model, dataloaders['valid'], criterion, gpu)
                print(
                    "Epoch: {}/{}... ".format(e+1, epochs),
                    "Training Loss: {:.4f}".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(loss/len(dataloaders['valid'])),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid']))
                )

            print("Iteration {} of epoch {}".format(steps, e+1))

            running_loss = 0

    print("Training completed via 'resume_training' function!")
    return model

def save_checkpoint(model, image_datasets, hidden_units, dropout, epochs, lr, file_path="checkpoint.pth"):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.hidden_units = hidden_units
    model.dropout = dropout
    model.epochs = epochs
    model.learning_rate = lr

    model.cpu()

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_units': model.hidden_units,
        'dropout': model.dropout,
        'epochs': model.epochs,
        'learning_rate': model.learning_rate,
    }

    torch.save(checkpoint, file_path)

def load_checkpoint(arch, file_path="checkpoint.pth"):
    available_archs = get_supported_archs()
    if arch not in available_archs:
        print("'{}' is not supported. Available options are vgg16 and alexnet.".format(arch))
    else:
        model = available_archs[arch][0]
        model_input = available_archs[arch][1]
        for param in model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(file_path)
        model.class_to_idx = checkpoint['class_to_idx']
        model.hidden_units = checkpoint['hidden_units']
        model.dropout = checkpoint['dropout']
        model.epochs = checkpoint['epochs']
        model.learning_rate = checkpoint['learning_rate']

        model = setup_nn(model, model_input, checkpoint['hidden_units'], checkpoint['dropout'])

        model.load_state_dict(checkpoint['state_dict'])
    
        return model

def process_image(image):
    im = Image.open(image)
    
    size = 256, 256
    im.thumbnail(size)
    
    # center crop
    width, height = im.size
    new_width, new_height = 224, 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    im = im.crop((left, top, right, bottom))
    
    npi = np.array(im)/255
    
    means = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    
    npi = (npi - means) / sd
    
    return npi.T

def unique_category_labels(file_path):
    labels_file = open(file_path, 'r')
    labels = set()

    for line in labels_file:
        labels.add(line.split('\t')[1].lower()[:-1])

    return labels

def get_label_to_folder_map():
    with open("label_to_folder_map.json", 'r') as f:
        label_to_folder_map = json.load(f)
    return label_to_folder_map

def predict(image_path, model, topk, gpu=0):
    img_array = process_image(image_path)
    processed_image = torch.from_numpy(img_array).type(torch.FloatTensor)
    processed_image = torch.unsqueeze(processed_image, 0)
    
    if gpu:
        device = determine_device(gpu)
        model.to(device)
        processed_image = processed_image.to(device)
    
    with torch.no_grad():
        output = model.forward(processed_image)
        
    probs = torch.exp(output)
    
    top_probs, top_labels = probs.topk(topk)

    top_probs = top_probs.cpu()
    top_labels = top_labels.cpu()
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labels = top_labels.detach().numpy().tolist()[0]
    # convert labels to int so that we can access idx_to_class keys
    top_labels_idx = [int(label) for label in top_labels]
    
    idx_to_class = {model.class_to_idx[k]: int(k)  for k in model.class_to_idx}

    name_to_cat = get_label_to_folder_map()

    # keys in "name_to_cat" should be values in cat_to_name
    # values in "name_to_cat" should be keys in cat_to_name
    cat_to_name = {}
    for key in name_to_cat:
        cat_to_name[name_to_cat[key]] = key
    
    top_idx_names = [cat_to_name[idx_to_class[lab]] for lab in top_labels_idx]
    
    top_categs = [cat_to_name[idx_to_class[lab]] for lab in top_labels]
    
    return top_probs, top_idx_names, top_categs
