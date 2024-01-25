import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#import the data
data = np.load('../Datasets/pathmnist.npz')

#split the data
train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

#gray out and reshape
train_images_gray = 0.299 * train_images[ :, :, :, 0] + 0.587 * train_images[ :, :, :, 1] + 0.114 * train_images[ :, :, :, 2]
train_images_gray = train_images_gray.reshape(89996, 1, 28, 28)
test_images_gray = 0.299 * test_images[:, :, :, 0] + 0.587 * test_images[:, :, :, 1] + 0.114 * test_images[:, :, :, 2]
test_images_gray= test_images_gray.reshape(7180, 1, 28, 28)
x_train = train_images_gray
y_train = train_labels.reshape(-1)
x_test = test_images_gray
y_test = test_labels.reshape(-1)

#transfer the format of data and implement preprocessing
y_train = torch.tensor(y_train, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)
x_train = torch.tensor(x_train, dtype = torch.float32)
x_train = x_train / 255.0
x_test = torch.tensor(x_test, dtype = torch.float32)
x_test = x_test / 255.0
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
x_train=transform(x_train)
x_test=transform(x_test)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128)

#build the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 72)
        self.fc3 = nn.Linear(72, 9) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32* 4* 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model =Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#train the model and print the accuracy
def evaluate_accuracy(data_loader, model):
    correct, total = 0, 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

num_epochs = 100
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = evaluate_accuracy(train_loader, model)
    test_acc = evaluate_accuracy(test_loader, model)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print(f'Epoch {epoch+1}, Training Accuracy: {train_acc}, Test Accuracy: {test_acc}')

#draw the picture
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


