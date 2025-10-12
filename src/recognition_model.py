import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import defaultdict

# change the input directory to the dataset path
root_dir = '/content/drive/MyDrive/Hand_gesture_dataset_V5'

gesture_classes = os.listdir(root_dir)
data = []
labels = []
label_to_int = {}

# Data Loading

i = 1
j = 0

for gestures in gesture_classes:
    class_dir = os.path.join(root_dir, gestures)
    print(f"Class {i}")
    i += 1
    for gesture_video in os.listdir(class_dir):
        video_path = os.path.join(class_dir, gesture_video)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Normalization without Rotation or Translation
        normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123])])

        normalized_frames = torch.stack([normalize(frame) for frame in frames])
        data.append(normalized_frames.permute(1, 0, 2, 3))  # Changing the channels and frames order

        # Normalization with only Rotation
        normalize_with_rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123])])

        rotated_frames = torch.stack([normalize_with_rotation(frame) for frame in frames])
        data.append(rotated_frames.permute(1, 0, 2, 3))  # Changing the channels and frames order

        # Normalization with only Translation
        normalize_with_translation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation of up to 10%
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123])])

        translated_frames = torch.stack([normalize_with_translation(frame) for frame in frames])
        data.append(translated_frames.permute(1, 0, 2, 3))  # Changing the channels and frames order

        if gestures not in label_to_int:
            label_to_int[gestures] = j
            j += 1

        labels.append(label_to_int[gestures])
        labels.append(label_to_int[gestures])  # Adding label for the rotated video
        labels.append(label_to_int[gestures])  # Adding label for the translated video

print("Gesture Classes:", labels)
print("label_to_int:", label_to_int)
print(len(labels))
number_of_classes = i

# Data Splitting
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = torch.stack(X_train_temp)
X_validation = torch.stack(X_validation)
X_test = torch.stack(X_test)

y_train = torch.tensor(y_train_temp)
y_validation = torch.tensor(y_validation)
y_test = torch.tensor(y_test)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_validation.shape)
print("Validation labels shape:", y_validation.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

print("Length of Training Data:", len(X_train))
print("Length of Validation Data:", len(X_validation))
print("Length of Testing Data:", len(X_test))

from torch.utils.data import DataLoader, TensorDataset

batch_size = 5

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

val_dataset = TensorDataset(X_validation, y_validation)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

import torch.nn as nn
import torchvision.models as models

class ResNet3D(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet3D, self).__init__()

        # Loading the pre-trained ResNet3D model
        self.resnet3d = models.video.r3d_18(pretrained=True)

        num_features = self.resnet3d.fc.in_features

        # adding dropout for regularization
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet3d(x)

num_classes = 17  # Adjust the classes
model = ResNet3D(num_classes)

model.eval()

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.00001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

model.to(device)
num_epochs = 45
patience = 5
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping! Best validation loss: {best_val_loss:.4f}")
            break
    scheduler.step(val_loss)

test_loss = 0.0
correct_test = 0
total_test = 0

model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

#saving the model
torch.save(model.state_dict(), 'gesture_model_T98%_.pth')

import matplotlib.pyplot as plt

# Plotting the training vs validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()

# Plotting the training vs validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the model's predictions
def get_predictions(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

predictions, true_labels = get_predictions(model, test_loader)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Plotting the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=True,
            xticklabels=list(label_to_int.keys()),
            yticklabels=list(label_to_int.keys()))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
