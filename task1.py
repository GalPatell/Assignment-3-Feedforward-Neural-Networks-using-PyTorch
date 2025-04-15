# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.use_deterministic_algorithms(True)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# images, labels = next(iter(train_loader))

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

test_error = []
train_error = []

def calculate_test_error() -> float:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    model.train()
    average_test_loss = test_loss / len(test_loader)
    return average_test_loss

def calculate_train_error() -> float:
    model.eval()
    train_loss = 0.0
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
    model.train()
    average_train_loss = train_loss / len(train_loader)
    return average_train_loss

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    train_error.append(calculate_train_error())
    test_error.append(calculate_test_error())
print(f'test error: {test_error[-1]}')

with torch.no_grad():
    correct = 0
    total = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified_images.append(images[i].cpu().reshape(28, 28))
                misclassified_labels.append(labels[i].cpu())
                misclassified_preds.append(predicted[i].cpu())

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Plot the training and test error
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_error, 'r', label='Training error')
plt.plot(epochs, test_error, 'b', label='Test error')
plt.title('Training and Test Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# Plot some misclassified images
fig, axs = plt.subplots(2, 5, figsize=(15, 8))
fig.suptitle('Misclassified Images', fontsize=16)

for i, ax in enumerate(axs.flat):
    if i < len(misclassified_images):
        ax.imshow(misclassified_images[i], cmap='gray')
        ax.set_title(f'True: {misclassified_labels[i]}\nPred: {misclassified_preds[i]}', fontsize=12)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
