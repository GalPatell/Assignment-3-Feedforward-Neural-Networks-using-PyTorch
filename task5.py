import os
# Fix OpenMP libraries conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE



# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Extract hidden features z_i and original features x_i for all x_i in the train set
hidden_features = []
original_features = []
labels = []

with torch.no_grad():
    for images, lbls in train_loader:
        images = images.reshape(-1, input_size).to(device)
        lbls = lbls.to(device)
        features = model.fc1(images)
        features = model.relu(features)
        hidden_features.append(features.cpu().numpy())
        original_features.append(images.cpu().numpy())
        labels.append(lbls.cpu().numpy())

hidden_features = np.concatenate(hidden_features)
original_features = np.concatenate(original_features)
labels = np.concatenate(labels)

# Use a smaller subset for t-SNE to reduce computation time
subset_size = 10000
idx = np.random.choice(len(hidden_features), subset_size, replace=False)
hidden_features_subset = hidden_features[idx]
original_features_subset = original_features[idx]
labels_subset = labels[idx]

# Apply t-SNE to hidden features
tsne_hidden = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
hidden_features_tsne = tsne_hidden.fit_transform(hidden_features_subset)

# Apply t-SNE to original features
tsne_original = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
original_features_tsne = tsne_original.fit_transform(original_features_subset)

# Plot t-SNE for hidden features
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(num_classes):
    idx = labels_subset == i
    plt.scatter(hidden_features_tsne[idx, 0], hidden_features_tsne[idx, 1], label=str(i), s=5)
plt.title('t-SNE of Hidden Features')
plt.legend()

# Plot t-SNE for original features
plt.subplot(1, 2, 2)
for i in range(num_classes):
    idx = labels_subset == i
    plt.scatter(original_features_tsne[idx, 0], original_features_tsne[idx, 1], label=str(i), s=5)
plt.title('t-SNE of Original Features')
plt.legend()

plt.show()