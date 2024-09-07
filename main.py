import csv
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Define a custom dataset
class CancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)  # Read the CSV file
        self.root_dir = root_dir  # Directory with all the images
        self.transform = transform  # Transformations for the images

    def __len__(self):
        return len(self.labels_df)  # Return total number of samples

    def __getitem__(self, idx):
        # Get image path and label
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.tif')
        image = Image.open(img_name).convert('RGB')  # Load the image and convert it to RGB
        label = self.labels_df.iloc[idx, 1]  # Get the label (0 or 1)

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label  # Return the image and label


# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # assuming input image size is (64,64)
        self.fc2 = nn.Linear(512, 1)  # output layer for binary classification

        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flattening
        x = x.view(-1, 128 * 8 * 8)  # adjust according to image size

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # binary classification output

        return x


class TestCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # return image and filename


def run_training():
    # Initialize the dataset
    train_dataset = CancerDataset(csv_file='data/train_labels.csv', root_dir='data/train', transform=transform)

    # Split the dataset into train and validation (80% train, 20% validation)
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # Create DataLoaders for train and validation
    train_loader = DataLoader(dataset=train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=64, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # Validation
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(running_val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Save model and plot training progress
    torch.save(model.state_dict(), 'simple_cnn_model.pth')

    # Plot the loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_validation_plots.png')
    plt.show()


def run_predicting():
    test_dataset = TestCancerDataset(root_dir='data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('simple_cnn_model.pth', weights_only=True))
    model.eval()

    # Make predictions and save to CSV
    predictions = []
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()

            for filename, predicted_label in zip(filenames, outputs):
                predictions.append([filename.replace('.tif', ''), float(predicted_label.item())])

    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])
        writer.writerows(predictions)

    print("Predictions saved to predictions.csv")


if __name__ == '__main__':
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(torch.cuda.get_device_name(0))  # Should return your GPU model
    else:
        print("Cuda unavailable")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    device = torch.device("cuda" if has_cuda else "cpu")
    print("Device:", device)

    # Define transformations
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    run_training()
    run_predicting()
