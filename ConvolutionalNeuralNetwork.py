# +------------------------------+
# | CNN for Paddy-Doctor-Dataset |
# +------------------------------+

# test accuracy = 56.73076923076923

import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

input_size = 65536*3
epochs = 10
learning_rate = 0.001
batch_size = 32

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
train_dataset = datasets.ImageFolder(root='data/paddy-doctor-diseases-small-400-split/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load the testing dataset
test_dataset = datasets.ImageFolder(root='data/paddy-doctor-diseases-small-400-split/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Example of iterating through the training data
for images, labels in train_loader:
    print(images.shape)  # Print the shape of the batch of images
    print(labels.shape)  # Print the shape of the batch of labels
    break  # Remove this line to iterate through the entire dataset

# Example of iterating through the testing data
for images, labels in test_loader:
    print(images.shape)  # Print the shape of the batch of images
    print(labels.shape)  # Print the shape of the batch of labels
    break  # Remove this line to iterate through the entire dataset


# Get first batch of training images and labels
train_images, train_labels = next(iter(train_loader))

# Display the batch of training images with labels
print("Training Images:")
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i][0])
plt.show()

# Get first batch of testing images and labels
test_images, test_labels = next(iter(test_loader))

# Display the batch of testing images with labels
print("Testing Images:")
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i][0])
plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*60*60, 1920)
        self.fc2 = nn.Linear(1920, 208)
        self.fc3 = nn.Linear(208, 13)

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 16*60*60)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out
    
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = NeuralNet().to(device)

# loss and the optimizer
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # Move images and labels to device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterian(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1}/{epochs}, step {i+1}/{n_total_steps}, loss {loss.item(): .4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

for images, lables in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

    # value, index
    _, predictions = torch.max(outputs, 1)
    n_samples += lables.shape[0]
    n_correct += (predictions == lables).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'accuracy = {acc}')


# # function to display batch of images not used in the code
# def show_batch(images, labels, classes, title="Images"):
#     fig, axes = plt.subplots(int(np.ceil(len(images) / 4)), 4, figsize=(16, 16))
#     fig.subplots_adjust(hspace=0.1, top=0.94)
#     axes = axes.flatten()

#     for i, (img, label) in enumerate(zip(images, labels)):
#         npimg = img.numpy()
#         ax = axes[i]
#         ax.imshow(np.transpose(npimg, (1, 2, 0)))
#         ax.set_title(classes[label], fontstyle='italic', fontsize='15', fontname='serif')
#         ax.axis('off')

#     for i in range(len(images), len(axes)):
#         axes[i].axis('off')

#     fig.suptitle(title, fontsize=20, color='red', fontstyle='normal', fontname='serif')
#     plt.show()