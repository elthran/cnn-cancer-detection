import os
import random
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
from main import CancerDataset, transform

# Load data
data = pd.read_csv('data/train_labels.csv')

# Display basic info and statistics
print(data.info())  # Shows column types and non-null values
print(data.describe())  # Shows basic statistics for numeric columns
print(data['label'].value_counts())  # Count of each class (0 or 1)

# Class distribution plot (Pie)
plt.figure(figsize=(6, 6))
data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Class Distribution (0: No Cancer, 1: Cancer)')
plt.show()

# Class distribution plot (Bar)
plt.figure(figsize=(6, 6))
data['label'].value_counts().plot(kind='bar', color=['blue', 'green'])
plt.title('Class Distribution (0: No Cancer, 1: Cancer)')
plt.ylabel('Count')
plt.show()

# Function to plot images with labels
def show_images_with_labels(images, labels, nrow=8):
    images = images / 2 + 0.5  # Unnormalize
    fig, axes = plt.subplots(2, nrow, figsize=(12, 6))
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i // nrow, i % nrow]  # Arrange in grid
        ax.imshow(img.permute(1, 2, 0))  # Rearrange dimensions for display
        ax.set_title(f'Label: {label.item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Function to filter and show images by label
def show_images_by_label(images, labels, label_value, n_samples=8):
    filtered_indices = [i for i, lbl in enumerate(labels) if lbl == label_value]
    random_indices = random.sample(filtered_indices, n_samples)
    sampled_images = images[random_indices]
    sampled_labels = labels[random_indices]
    return sampled_images, sampled_labels

# Load a few samples from the dataset
train_dataset = CancerDataset(csv_file='data/train_labels.csv', root_dir='data/train', transform=transform)
loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Get a batch of images and labels
images, labels = next(iter(loader))

# Sample 8 images where label == 1 and 8 images where label == 0
images_label_0, labels_label_0 = show_images_by_label(images, labels, label_value=0, n_samples=8)
images_label_1, labels_label_1 = show_images_by_label(images, labels, label_value=1, n_samples=8)

# Concatenate and display
images_concat = torch.cat((images_label_0, images_label_1), 0)
labels_concat = torch.cat((labels_label_0, labels_label_1), 0)
show_images_with_labels(images_concat, labels_concat)

# Distribution of pixel intensities for a few sample images
sample_images, _ = next(iter(loader))  # Take one batch of images
sample_images = sample_images.view(-1).numpy()  # Flatten the images
plt.hist(sample_images, bins=50, color='blue', alpha=0.7)
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()