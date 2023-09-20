import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()

def plot_predictions_vs_true(model, dataloader):
    true_values = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            true_values.extend(labels.numpy())
            predictions.extend(outputs.numpy())

    plt.scatter(true_values, predictions, alpha=0.6)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Values')
    plt.grid(True)
    plt.show()

def plot_error_distribution(model, dataloader):
    true_values = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            true_values.extend(labels.numpy())
            predictions.extend(outputs.numpy())

    errors = [pred - true for pred, true in zip(predictions, true_values)]
    plt.hist(errors, bins=50, alpha=0.6)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data from CSV
    data = pd.read_csv("your_file.csv")
    features = data[["temperature", "current", "voltage"]].values
    targets = data["state_of_charge"].values

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # DataLoader
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=len(features_tensor))  # Batch size = entire dataset

    # Visualize for a trained model
    model = torch.load("your_saved_model.pth")  # Assuming you've saved your model in this filename

    plot_predictions_vs_true(model, dataloader)
    plot_error_distribution(model, dataloader)
