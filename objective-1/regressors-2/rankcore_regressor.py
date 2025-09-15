import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Function to calculate PLCC loss (maximize PLCC, so minimize negative PLCC)
def plcc_loss(y_true, y_pred):
    # Ensure inputs are 1D
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    # Center the data
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    # Calculate numerator and denominator
    numerator = torch.sum(y_true_centered * y_pred_centered)
    denominator = torch.sqrt(torch.sum(y_true_centered**2) * torch.sum(y_pred_centered**2))

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    plcc_val = numerator / (denominator + epsilon)

    # We want to maximize PLCC, so our loss is the negative of PLCC
    return -plcc_val

# Function to calculate SRCC loss
def srcc_loss(y_true, y_pred):
    # Rank the true and predicted values
    y_true_rank = torch.argsort(torch.argsort(y_true))
    y_pred_rank = torch.argsort(torch.argsort(y_pred))

    # Calculate the squared differences between the ranks
    d_squared = (y_true_rank.float() - y_pred_rank.float())**2

    # Number of elements
    n = len(y_true)

    # Spearman's rank correlation coefficient formula
    srcc_val = 1 - (6 * torch.sum(d_squared)) / (n * (n**2 - 1))

    # We want to maximize SRCC, so our loss is the negative of SRCC
    return -srcc_val

class RankCORERegressor(nn.Module):
    def __init__(self, input_size, hidden_size, loss_type='plcc'):
        super(RankCORERegressor, self).__init__()
        self.loss_type = loss_type
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def fit(self, X, y, n_epochs, learning_rate, batch_size):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            permutation = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X[indices], y[indices]

                # Ensure batch is not too small for loss calculation
                if len(batch_X) < 2:
                    continue
                
                optimizer.zero_grad()
                y_pred = self.forward(batch_X)

                if self.loss_type == 'plcc':
                    loss = plcc_loss(batch_y, y_pred)
                elif self.loss_type == 'srcc':
                    loss = srcc_loss(batch_y, y_pred)
                else:
                    raise ValueError("loss_type must be 'plcc' or 'srcc'")

                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.forward(X_tensor)
        return y_pred.squeeze().numpy()

class RankCORERegressorWrapper:
    """Wrapper class to make the PyTorch model compatible with the existing scikit-learn training pipeline."""
    def __init__(self, n_epochs=100, learning_rate=0.01, loss_type='plcc', hidden_size=128, batch_size=16):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        input_size = X.shape[1]
        self.model = RankCORERegressor(input_size, self.hidden_size, self.loss_type)
        self.model.fit(X, y, self.n_epochs, self.learning_rate, self.batch_size)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'loss_type': self.loss_type,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self