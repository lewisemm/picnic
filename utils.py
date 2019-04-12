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
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    
    dirs = {
        "train": train_dir, 
        "valid": valid_dir, 
        "test" : test_dir
    }
    
    return image_datasets, dataloaders

def get_supported_archs():
    return {
        "vgg16": (models.vgg16(pretrained=True), 25088),
        # "alexnet": (models.alexnet(pretrained=True), 9216)
    }
        
def setup_nn(model, model_input, hidden_units, dropout=0.4):
    # maps the model string to the torchvision model and it's corresponding input
    # features at the classifier

    for param in model.parameters():
        param.requires_grad = False

    flower_classifier = None
    dropout = nn.Dropout(p=dropout)

    if len(hidden_units) <= 2:
        flower_classifier = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(model_input, hidden_units[0])),
                ("relu1", nn.ReLU()),
                ("dropout1", dropout),
                ("fc2", nn.Linear(hidden_units[0], hidden_units[-1])),
                ("relu2", nn.ReLU()),
                ("dropout2", dropout),
                ("fc3", nn.Linear(hidden_units[-1], 102)),
                ("output", nn.LogSoftmax(dim=1))
            ])
        )
    elif len(hidden_units) > 2:
        fields = [
            ("fc1", nn.Linear(model_input, hidden_units[0])),
            ("relu1", nn.ReLU()),
            ("dropout1", dropout)
        ]
        previous_input = hidden_units[0]
        for index, unit in enumerate(hidden_units[1:]):
            fields.append(
                ("fc{}".format(index + 2), nn.Linear(previous_input, unit))
            )
            fields.append(
                ("relu{}".format(index + 2), nn.ReLU())
            )
            fields.append(
                ("dropout{}".format(index + 2), dropout)
            )
            previous_input = unit
        fields.append(
            ("fc{}".format(len(hidden_units) + 1), nn.Linear(hidden_units[-1], 26))
        )
        fields.append(
            ("output", nn.LogSoftmax(dim=1))
        )

        flower_classifier = nn.Sequential(OrderedDict(fields))
    
    
    model.classifier = flower_classifier
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
            steps += 0 
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

            running_loss = 0

    print("Training Complete!")
    return model

def test_network(model, criterion, test_dataloader):
    device = determine_device()
    model.to(device)
    
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)

            test_loss += criterion(output, labels).item()

            probs = torch.exp(output)
            equality = (labels.data == probs.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            print("Testing Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                    "Testing Accuracy: {:.3f}".format(accuracy/len(test_dataloader)))
            
    print("Testing Complete!")

def save_checkpoint(model, image_datasets, hidden_units, file_path="checkpoint.pth"):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.hidden_units = hidden_units

    model.cpu()

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_units': model.hidden_units
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

        model = setup_nn(model, model_input, checkpoint['hidden_units'])

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

def predict(image_path, model, topk, category_json_path, gpu=0):
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
    
    idx_to_class = {model.class_to_idx[k]: k  for k in model.class_to_idx}

    cat_to_name = get_category_json(category_json_path)
    
    top_idx_names = [cat_to_name[idx_to_class[lab]] for lab in top_labels_idx]
    
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labels]
    
    return top_probs, top_idx_names, top_flowers