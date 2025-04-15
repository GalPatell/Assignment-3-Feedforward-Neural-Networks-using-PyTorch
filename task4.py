import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_sizes = [500 , 750]
num_classes = 10
num_epochs = 5
batch_sizes = [50,100,200]
learning_rates = [1e-4, 1e-5]
validation_size = 10000

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)


# MNIST dataset
full_train_dataset = torchvision.datasets.MNIST(
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

train_size = len(full_train_dataset) - validation_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size])

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

test_error_best = []
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            model = NeuralNet(input_size, hidden_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_errors = []
            test_errors = []

            def calculate_error(loader):
                model.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for images, labels in loader:
                        images = images.reshape(-1, input_size).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()
                model.train()
                average_loss = total_loss / len(loader)
                return average_loss
            
            best_val_error = float('inf')
            best_test_error = float('inf')

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
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

                val_error = calculate_error(val_loader)
                test_error = calculate_error(test_loader)
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_test_error = test_error

            test_error_best.append(best_test_error)


for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes: 
            print(f'Batch size: {batch_size}, Hidden size: {hidden_size}, Learning rate: {learning_rate}, Test error: {test_error_best.pop(0)}')        
                

                


# images, labels = next(iter(train_loader))



# Fully connected neural network
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out



# def train_and_evaluate(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.use_deterministic_algorithms(True)

#     model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     val_errors = []
#     test_errors = []

#     def calculate_error(loader):
#         model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for images, labels in loader:
#                 images = images.reshape(-1, input_size).to(device)
#                 labels = labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item()
#         model.train()
#         average_loss = total_loss / len(loader)
#         return average_loss
    
#     best_val_error = float('inf')
#     best_test_error = float('inf')

#     # Train the model
#     total_step = len(train_loader)
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.reshape(-1, input_size).to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

#         val_error = calculate_error(val_loader)
#         test_error = calculate_error(test_loader)
#         val_errors.append(val_error)
#         test_errors.append(test_error)

#         if val_error < best_val_error:
#             best_val_error = val_error
#             best_test_error = test_error

#     return val_errors, test_errors, best_test_error, best_val_error

# Main execution with different seeds

# all_val_errors = []
# all_test_errors = []
# best_test_errors = []
# best_val_errors = []

# plt.figure(figsize=(10, 6))


# val_errors, test_errors, best_test_error, best_val_error = train_and_evaluate(seed)
# all_val_errors.append(val_errors)
# all_test_errors.append(test_errors)
# best_test_errors.append(best_test_error)
# best_val_errors.append(best_val_error)
# plt.plot(val_errors, label=f'Seed {seed} - Validation Error')
# plt.plot(test_errors, label=f'Seed {seed} - Test Error', linestyle='--')

# plt.title('Validation and Test Errors During Training')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.legend()
# plt.show()

# # Calculate mean and standard deviation of the best test errors
# mean_best_test_error = np.mean(best_test_errors)
# std_best_test_error = np.std(best_test_errors)
# mean_best_val_error = np.mean(best_val_errors)
# std_best_val_error = np.std(best_val_errors)
# print(f'Mean best test error: {mean_best_test_error:.4f}')
# print(f'Standard deviation of best test error: {std_best_test_error:.4f}')
# print(f'Mean best validation error: {mean_best_val_error:.4f}')
# print(f'Standard deviation of best validation error: {std_best_val_error:.4f}')


