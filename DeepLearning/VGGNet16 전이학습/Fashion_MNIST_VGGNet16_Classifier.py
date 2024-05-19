import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import FashionMNIST
from torchvision.models.vgg import vgg16
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

import tqdm
from torch.optim.adam import Adam

import numpy as np
import matplotlib.pyplot as plt
import time


""" Set pretrained model """
model = vgg16(pretrained=True)

# classifier
fc = nn.Sequential(
    nn.Linear(512*7*7, 4096),
    nn.ReLU(),
    nn.Dropout(), # dropout layer 정의
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10, bias=False),
    nn.Softmax(dim=1)
)
model.classifier = fc

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(model)

summary(model,input_size=(3, 224, 224))


""" Dataset preproceesing & load """
transforms = Compose([
    Grayscale(num_output_channels=3),
    Resize(224),
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

training_data = FashionMNIST(root="./data", train=True, download=True, transform=transforms)
test_data = FashionMNIST(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# freezing pretrained parameter
for name, param in model.features.named_parameters():
    param.requires_grad = False

params_to_update = []
for name, param in model.classifier.named_parameters():
    param.requires_grad = True
    params_to_update.append(param)


""" Training """
lr = 1e-4
loss_fn = nn.CrossEntropyLoss()
optim = Adam(params=params_to_update, lr=lr)

losses = []
size = len(train_loader.dataset)

for epoch in range(2):
    start_time = time.time()
    epoch_loss = 0.0
    
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        loss = loss_fn(preds, label.to(device))
        loss.backward()
        optim.step()
        
        epoch_loss += loss.item() * data.size(0)
     
        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

    epoch_loss = epoch_loss / size
    losses.append(epoch_loss)
    
    calculation_time = time.time() - start_time
    print(f"Calculation time on training set : {calculation_time:0.3f} sec")

# draw loss function trend
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# save parameters
torch.save(model.state_dict(), "FashionMNIST_pretrained.pth")

# # load parameters
# model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))


""" Validation """
num_corr = 0
correct_samples = []
incorrect_samples = []

model.eval()
with torch.no_grad():
    start_time = time.time()
    iterator = tqdm.tqdm(test_loader)
    for data, label in iterator:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
        
        for i in range(len(preds)):
            if preds[i] == label[i]:
                correct_samples.append((data[i], preds[i]))
            else:
                incorrect_samples.append((data[i], preds[i], label[i]))
    
    validation_time = time.time() - start_time
    print(f"Calculation time on validation set: {validation_time:.3f} sec")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Correctly classified samples:")
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(correct_samples[i][0].permute(1, 2, 0))
    plt.title("Predicted: " + class_names[correct_samples[i][1]])
    plt.axis('off')
plt.show()

print("Incorrectly classified samples:")
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(incorrect_samples[i][0].permute(1, 2, 0))
    plt.title("Predicted: " + class_names[incorrect_samples[i][1]] + ", Actual: " + class_names[incorrect_samples[i][2]])
    plt.axis('off')
plt.show()

    
print(f"Accuracy:{num_corr/len(test_data)}")