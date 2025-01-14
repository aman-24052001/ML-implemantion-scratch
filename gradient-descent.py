import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from typing import Tuple, List

class GradientDescent:
    def __init__(self, learning_rate: float = 0.01, n_iterations: float = 1000):
        """
        Initialize gradient descent optimizer
        
        Args:
            learning_rate: Step size for parameter updates
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def initialize_parameters(self, n_features: int) -> None:
        """Initialize model parameters randomly"""
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
    def compute_predictions(self, X: np.ndarray) -> np.ndarray:
        """Forward pass to compute predictions"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias"""
        m = len(y_true)
        dw = -(2/m) * np.dot(X.T, (y_true - y_pred))
        db = -(2/m) * np.sum(y_true - y_pred)
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Train the model using gradient descent
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            List of loss values during training
        """
        # Initialize parameters
        self.initialize_parameters(X.shape[1])
        
        # Gradient descent loop
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.compute_predictions(X)
            
            # Compute loss
            current_loss = self.compute_loss(y, y_pred)
            self.loss_history.append(current_loss)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self.loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return self.compute_predictions(X)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and R² score"""
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, r2

def plot_results(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, loss_history: List[float]) -> None:
    """Plot training results and loss curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot predictions vs actual
    ax1.scatter(X[:, 0], y, color='blue', label='Actual')
    ax1.scatter(X[:, 0], y_pred, color='red', label='Predicted')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Target')
    ax1.set_title('Predictions vs Actual')
    ax1.legend()
    
    # Plot loss history
    ax2.plot(loss_history)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss Over Time')
    
    plt.tight_layout()
    plt.show()

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = GradientDescent(learning_rate=0.01, n_iterations=1000)
loss_history = model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate model
train_mse, train_r2 = evaluate_model(y_train, train_predictions)
test_mse, test_r2 = evaluate_model(y_test, test_predictions)

# Display results
metrics = [
    ["Training", train_mse, train_r2],
    ["Testing", test_mse, test_r2]
]
headers = ["Dataset", "MSE", "R² Score"]
print("\nModel Performance Metrics:")
print(tabulate(metrics, headers=headers, floatfmt=".4f", tablefmt="grid"))

# Plot results
plot_results(X_train, y_train, train_predictions, loss_history)

# Print model parameters
print("\nFinal Model Parameters:")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")

# Model Analysis
if abs(train_r2 - test_r2) < 0.1:
    conclusion = "✅ Good generalization: Model performs similarly on training and test data"
else:
    conclusion = "⚠️ No improvement: The model failed to learn from the training data or overfitted"

print("\nModel Analysis:")
print(conclusion)
