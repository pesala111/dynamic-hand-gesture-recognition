"""
Gesture recognition model training and evaluation pipeline.
Builds and trains a ResNet-3D (r3d_18) model on the hand gesture dataset
with data augmentation (rotation, translation), trains with early stopping,
and evaluates on a held-out test set.

Usage:
    python recognition_model.py --root_dir /path/to/Hand_gesture_dataset
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.video import R3D_18_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-3D gesture recognition model.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the root dataset directory containing one sub-folder per gesture class.",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Training batch size (default: 5).")
    parser.add_argument("--num_epochs", type=int, default=45, help="Maximum number of training epochs (default: 45).")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs (default: 5).")
    parser.add_argument("--output_model", type=str, default="gesture_model.pth", help="Filename for the saved model weights (default: gesture_model.pth).")
    return parser.parse_args()


def load_data(root_dir):
    """Load video frames from root_dir, apply augmentations, and return data/label tensors."""
    gesture_classes = sorted(os.listdir(root_dir))
    data = []
    labels = []
    label_to_int = {}

    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123]),
    ])
    normalize_with_rotation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123]),
    ])
    normalize_with_translation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123]),
    ])

    for idx, gesture in enumerate(gesture_classes):
        label_to_int[gesture] = idx
        class_dir = os.path.join(root_dir, gesture)
        print(f"Loading class {idx + 1}/{len(gesture_classes)}: {gesture}")

        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if not frames:
                continue

            # Original + two augmented copies
            data.append(torch.stack([normalize(f) for f in frames]).permute(1, 0, 2, 3))
            data.append(torch.stack([normalize_with_rotation(f) for f in frames]).permute(1, 0, 2, 3))
            data.append(torch.stack([normalize_with_translation(f) for f in frames]).permute(1, 0, 2, 3))
            labels.extend([idx, idx, idx])

    number_of_classes = len(gesture_classes)
    print(f"Loaded {len(labels)} samples across {number_of_classes} classes.")
    print("label_to_int:", label_to_int)
    return data, labels, label_to_int, number_of_classes


def build_dataloaders(data, labels, batch_size):
    """Split data into train/val/test sets and return DataLoaders."""
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    X_train = torch.stack(X_train_temp)
    X_validation = torch.stack(X_validation)
    X_test = torch.stack(X_test)
    y_train = torch.tensor(y_train_temp)
    y_validation = torch.tensor(y_validation)
    y_test = torch.tensor(y_test)

    print(f"Train: {X_train.shape}, Val: {X_validation.shape}, Test: {X_test.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class ResNet3D(nn.Module):
    """ResNet-3D wrapper with a configurable classification head and dropout."""

    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.resnet3d = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.resnet3d(x)


def train(model, train_loader, val_loader, device, num_epochs, patience, output_model):
    """Train model with early stopping; save best weights to output_model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, lbls in train_loader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += lbls.size(0)
            correct_train += (predicted == lbls).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # --- Validation phase ---
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, lbls in val_loader:
                inputs, lbls = inputs.to(device), lbls.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, lbls)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += lbls.size(0)
                correct_val += (predicted == lbls).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered. Best val loss: {best_val_loss:.4f}")
                break

        scheduler.step(val_loss)

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate(model, test_loader, device):
    """Evaluate model on test set and return loss, accuracy, predictions, and true labels."""
    criterion = nn.CrossEntropyLoss()
    test_loss, correct_test, total_test = 0.0, 0, 0
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, lbls in test_loader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += lbls.size(0)
            correct_test += (predicted == lbls).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training vs validation loss and accuracy curves."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(predictions, true_labels, label_to_int):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, predictions)
    class_names = list(label_to_int.keys())
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm, annot=True, cmap='Blues', fmt='g', cbar=True,
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    # Load data
    data, labels, label_to_int, number_of_classes = load_data(args.root_dir)

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(data, labels, args.batch_size)

    # Build model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet3D(num_classes=number_of_classes).to(device)

    # Train
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs,
        patience=args.patience,
        output_model=args.output_model,
    )

    # Load best weights for evaluation
    model.load_state_dict(torch.load(args.output_model))

    # Evaluate
    _, _, predictions, true_labels = evaluate(model, test_loader, device)

    # Visualise
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(predictions, true_labels, label_to_int)
