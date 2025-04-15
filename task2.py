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

def train_and_evaluate(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

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

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    test_errors = []

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
        test_errors.append(calculate_test_error())
    return test_errors

# Main execution with different seeds
seeds = [0, 50, 100, 300, 1000]
all_test_errors = []

plt.figure(figsize=(10, 6))

for seed in seeds:
    test_error = train_and_evaluate(seed)
    all_test_errors.append(test_error)
    plt.plot(test_error, label=f'Seed {seed}')

plt.title('Test Errors During Training')
plt.xlabel('Epoch')
plt.ylabel('Test Error')
plt.legend()
plt.show()

# Calculate mean and standard deviation of the final test errors
final_test_errors = [errors[-1] for errors in all_test_errors]
mean_final_error = np.mean(final_test_errors)
std_final_error = np.std(final_test_errors)

print(f'Mean final test error: {mean_final_error:.4f}')
print(f'Standard deviation of final test error: {std_final_error:.4f}')


