import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def imshow(img):
    """Wyświetlanie obrazka po odnormalizowaniu."""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)

        self.dropout_conv = nn.Dropout(0.2)

        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_output = self._forward_conv_layers(dummy_input)
        self.flattened_size = dummy_output.numel()


        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

        self.dropout_fc = nn.Dropout(0.6)

    def _forward_conv_layers(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout_conv(x) 
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout_conv(x)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1) 
        x = self.dropout_fc(F.relu(self.fc1(x)))  
        x = self.dropout_fc(F.relu(self.fc2(x)))  
        x = self.fc3(x) 
        return x


def train_model(data_dir, batch_size=4, num_epochs = 15):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    valset = ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {running_loss/len(trainloader):.4f}, '
              f'Val Loss: {val_loss/len(valloader):.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')
    
    print('Finished Training')
    torch.save(net.state_dict(), './bearSGD_net.pth')


def main():
    data_dir = "G:/PSI zaliczenie/Zad2/data"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = testset.classes
    print("Classes:", classes)

    net = Net()
    net.load_state_dict(torch.load('./bearSGD_net.pth'))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    data_dir = "G:/PSI zaliczenie/Zad2/data"
    train_model(data_dir=data_dir, batch_size=4, num_epochs=15)
    main()
