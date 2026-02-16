# trains model to guess digits

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

# LeNet that serves as perceptron
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 120)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(120, 10)
        #self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        #x = self.sm(x)
        return x


def create_rand_mask(n, m):
    if n > m * m:
        print("n should be less than or equal to m*m, None returned")
        return None

    matrix = [[0] * m for _ in range(m)]
    indices = random.sample(range(m * m), n)

    for idx in indices:
        row = idx // m
        col = idx % m
        matrix[row][col] = 1

        tensor = np.expand_dims(matrix, axis=0)

    return tensor

def train_perceptron(epochs=3, random_masking=False, saving=True):


    # Declare transform to convert raw data to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading Data and splitting it into train and validation data
    train = datasets.MNIST('', train=True, transform=transform, download=True)
    train, valid = random_split(train, [50000, 10000])

    # Create Dataloader of the above tensor with batch size = 32
    trainloader = DataLoader(train, batch_size=32)
    validloader = DataLoader(valid, batch_size=32)

    # Building Our Mode
    model = Perceptron()
    if torch.cuda.is_available():
        model = model.cuda()

    # Declaring Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training with Validation
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        #model.train()
        for data, labels in trainloader:

            if random_masking:
                # adding unexplored pixels
                random_mask = torch.randint(0, 2, size=data.size())
                data = data * random_mask

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        correct_randomized = 0
        total_randomized = 0
        for data, labels in validloader:

            if random_masking:
                # adding unexplored pixels
                random_mask = torch.randint(0, 2, size=data.size())
                #data = data * random_mask

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate Loss
            valid_loss += loss.item()

            #####
            _, predicted_randomized = torch.max(target.data, 1)
            total_randomized += labels.size(0)
            correct_randomized += (predicted_randomized == labels).sum().item()

        print(
            f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t - validation Loss: {valid_loss / len(validloader)}')
        accuracy_randomized = correct_randomized / total_randomized
        print(f'Accuracy on images: {accuracy_randomized * 100:.2f}%')
        if saving:
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss

                # Saving State Dict
                torch.save(model.state_dict(), 'trained_LeNet.pth')
    return model

def test_perceptron(model):
    # Define the test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='', train=False, transform=transform, download=True)
    indices = torch.randperm(len(test_dataset))[:100]
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    model.eval()

    #model = network  # I replace model with validated model called network

    # Evaluate the model on the test set
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy on the loaded model: {accuracy * 100:.2f}%')

    # Test the model on images with random half of the pixels set to 0
    model.eval()
    correct_randomized = 0
    total_randomized = 0


    with torch.no_grad():
        for images, labels in test_loader:
            # Randomly set half of the pixels to 0
            #random_mask = torch.randint(0, 2, size=images.size())
            npix = 300
            random_mask = torch.zeros(size=images.size())
            number_of_samples = images.shape[0]

            for i in range(number_of_samples):
                msk = create_rand_mask(npix, 28)
                random_mask[i, 0, :, :] = torch.from_numpy(msk)
                #plt.figure()
                #plt.imshow(msk[0,:,:], cmap="gray")
                #plt.show()


            images_randomized = images * random_mask

            outputs_randomized = model(images_randomized)
            _, predicted_randomized = torch.max(outputs_randomized.data, 1)
            total_randomized += labels.size(0)
            correct_randomized += (predicted_randomized == labels).sum().item()

    accuracy_randomized = correct_randomized / total_randomized
    print(f'Accuracy on images with random npixels set to 0: {accuracy_randomized * 100:.2f}%')


if __name__ == "__main__":
    mdl = train_perceptron()
    test_perceptron(mdl)
