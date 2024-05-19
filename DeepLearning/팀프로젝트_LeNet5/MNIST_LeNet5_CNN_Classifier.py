import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import time


# MNIST dataset load
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 모델 정의
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0, stride=1)
        
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84,out_features=10)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()


    def forward(self, x):
        # Building Block 1
        x = self.conv1(x)   # (6, 28, 28)
        x = self.relu(x)    
        x = self.pool(x)    # (6, 14, 14)
        
        # Building Block 2
        x = self.conv2(x)   # (16, 10, 10)
        x = self.relu(x)
        x = self.pool(x)    # (16, 5, 5)
        
        # Building Block 3
        x = self.conv3(x)   # (120, 1, 1)
        x = self.relu(x)
                           
        # Serialization for 2D image * channels                           
        x = self.flatten(x) # (120, 1)
                            
        # Fully connected layers
        x = self.linear1(x)
        x = self.relu(x)
        
        # output layer
        x = self.linear2(x)
        x = self.log_softmax(x)
        
        return x


model = LeNet5().to(device)
print(f"model : {model}")
print("\n################################\n")

# 모델 매개변수 최적화
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 모델 학습
def train(dataloader, model, loss_fn, optimizer):
    start_time = time.time()
    
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    calculation_time = time.time() - start_time
    print(f"\n* Calculation time on training set\n : {calculation_time:0.3f} sec\n")
    
    return loss


# 모델 성능 평가 및 정확하게 분류한 10개의 랜덤 sample 출력
def test(dataloader, model, classes):
    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    correct_samples = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            _, predicted = torch.max(outputs, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()
            
            # 정확하게 분류된 샘플 저장
            for i in range(len(predicted)):
                if predicted[i] == y[i]:
                    correct_samples.append((X[i], predicted[i]))
    
    accuracy = 100 * correct_predictions / total_samples
    print(f"* Accuracy on test dataset\n : {accuracy:.2f}%")
    
    # 정확하게 분류된 샘플 중에서 랜덤하게 선택하여 출력
    print("\n* Correctly classified samples")
    for i in range(10):
        sample = correct_samples[np.random.randint(0, len(correct_samples))]
        image, label = sample
        
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"({i + 1}) Predicted: {classes[label]}")
        plt.show()
        

# test dataset에서 잘못 분류된 샘플 출력
def print_misclassified_samples(dataloader, model, classes):
    model.eval()
    misclassified_samples = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            # 잘못 분류된 샘플 식별 및 저장
            for i in range(len(predicted)):
                if predicted[i] != y[i]:
                    misclassified_samples.append((X[i], predicted[i], y[i]))
    
    # 잘못 분류된 샘플 출력
    print("\n* Misclassified samples")
    num_samples = len(misclassified_samples)
    num_subplots = 4*4
    num_images = int(np.ceil(num_samples/num_subplots))
    
    for idx in range(num_images):
        fig, axes = plt.subplots(4, 4, figsize=(9, 9))
        start_idx = idx*num_subplots
        end_idx = min((idx + 1)*num_subplots, num_samples)
        
        for i in range(start_idx, end_idx):
            row_idx = (i - start_idx)//4
            col_idx = (i - start_idx)%4
            ax = axes[row_idx, col_idx]
            image, predicted_label, true_label = misclassified_samples[i]
            
            # 이미지 출력
            image_np = image.squeeze().cpu().numpy()
            ax.imshow(image_np, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Predicted: {classes[predicted_label]}, True: {classes[true_label]}")

        plt.tight_layout()
        plt.show()


# 에폭에 따른 결과
error_train = []
epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    error = train(train_dataloader, model, loss_fn, optimizer)
    error_train.append(error)

# training set에 대한 cost function 추세
plt.figure(1, figsize=(5, 5))
plt.plot([e.detach().numpy() for e in error_train], 'black', label='cost function')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Trend of Cost Function for Training Set')
plt.show()
print("Done!\n")

# 모델 저장
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth\n")

# inference
classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

# # 가중치 파일이 존재하는 경우
# model.load_state_dict(torch.load('model.pth'))

test(test_dataloader, model, classes)

print_misclassified_samples(test_dataloader, model, classes)
