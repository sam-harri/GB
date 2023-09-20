import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class BatteryStatePredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=300, dropout_probability=0.2):
        super(BatteryStatePredictor, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_probability)

        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Consider only the last output

        # Dropout
        out = self.dropout(lstm_out)
        
        # Batch normalization
        out = self.batch_norm(out)

        # Fully connected layer
        out = self.fc(out)

        return out

    def train_model(self, dataloader, criterion, optimizer, epochs=10):
        # Training loop
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


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

    # Define model, loss and optimizer
    model = BatteryStatePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train_model(dataloader, criterion, optimizer, epochs=50)
