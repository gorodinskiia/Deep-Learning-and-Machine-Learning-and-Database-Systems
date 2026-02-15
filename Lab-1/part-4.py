import torch  # PyTorch library for tensor computations
import math   # Math library for sqrt
import matplotlib.pyplot as plt  # Plotting library
import random  # Python random module (not strictly needed here)

# python3 part-4.py

def create_linear_dataset():
    # Create a simple 2D linearly separable dataset
    # Points along x-axis and y-axis to make a linearly separable case
    points = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0]
    ])
    labels = torch.tensor([1, 0, 0, 1]).unsqueeze(1)  # Assign linearly separable labels
    return points, labels  # Return dataset

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))  # Sigmoid activation

def sigmoid_derivative(x):
    s = sigmoid(x)  # Compute sigmoid
    return s * (1 - s)  # Its derivative

class TwoLayerNN:
    def __init__(self, d_input, d_hidden=2, lr=0.1, init_scheme="gaussian"):
        self.lr = lr  # Learning rate
        self.d_input = d_input  # Input dimension
        self.d_hidden = d_hidden  # Hidden dimension
        
        if init_scheme == "gaussian":  # Baseline Gaussian initialization
            self.w1 = torch.randn(d_hidden, d_input) / math.sqrt(d_input)  # Gaussian scaled
            self.w2 = torch.randn(d_hidden, 1) / math.sqrt(d_hidden)  # Gaussian scaled
        elif init_scheme == "uniform":  # Uniform initialization
            self.w1 = (torch.rand(d_hidden, d_input) - 0.5) * 2 / (10 * d_input)  # Uniform in [-1/(10*d_input), 1/(10*d_input)]
            self.w2 = (torch.rand(d_hidden, 1) - 0.5) * 2 / (10 * math.sqrt(d_hidden))  # Uniform in [-1/(10*sqrt(d_hidden)), 1/(10*sqrt(d_hidden))]
        elif init_scheme == "constant":  # Constant 1.0 initialization
            self.w1 = torch.ones(d_hidden, d_input)  # All weights = 1.0
            self.w2 = torch.ones(d_hidden, 1)  # All weights = 1.0
        else:
            raise ValueError("Unknown initialization scheme")
        # Initialize biases to zero
        self.b1 = torch.zeros(d_hidden, 1)  # Hidden biases
        self.b2 = torch.zeros(1, 1)  # Output bias

    def forward(self, x):
        self.p1 = self.w1 @ x.unsqueeze(1) + self.b1  # Linear combination for hidden layer
        self.o1 = sigmoid(self.p1)  # Hidden activations
        self.p2 = self.w2.T @ self.o1 + self.b2  # Linear combination for output
        self.y_hat = sigmoid(self.p2)  # Output activation
        return self.y_hat  # Return output
    
    def backward(self, x, y):
        y = torch.tensor([[y]], dtype=torch.float32)  # Target as tensor (1,1)
        delta2 = (self.y_hat - y) * sigmoid_derivative(self.p2)  # Output layer error
        dw2 = self.o1 @ delta2.T  # Gradient for output weights
        db2 = delta2  # Gradient for output bias
        delta1 = (self.w2 * delta2) * sigmoid_derivative(self.p1)  # Hidden layer error
        dw1 = delta1 @ x.unsqueeze(0)  # Gradient for hidden weights
        db1 = delta1  # Gradient for hidden biases
        self.w2 -= self.lr * dw2  # Update output weights
        self.b2 -= self.lr * db2  # Update output bias
        self.w1 -= self.lr * dw1  # Update hidden weights
        self.b1 -= self.lr * db1  # Update hidden biases

    def predict(self, x):
        return 1 if self.forward(x).item() >= 0.5 else 0  # Threshold at 0.5

def train_until_convergence(points, labels, lr=0.1, epochs=500, init_scheme="gaussian"):
    """
    Train model until it reaches 100% accuracy and return the first epoch it happens (convergence time)
    """
    model = TwoLayerNN(d_input=2, lr=lr, init_scheme=init_scheme)  # Initialize model
    for epoch in range(1, epochs + 1):  # Loop through epochs
        correct = 0  # Count correct predictions
        for i in range(len(points)):
            x = points[i]  # Input
            y = labels[i].item()  # Target

            pred = model.predict(x)  # Make prediction
            if pred == y:  # If correct
                correct += 1

            model.backward(x, y)  # Update weights

        if correct == len(points):  # If all predictions correct
            return epoch  # Return convergence epoch

    return epochs  # If never converged, return max epoch

def plot_convergence_box(convergence_gaussian, convergence_uniform, convergence_constant):
    """
    Plot boxplot of convergence times for Gaussian and Uniform schemes
    Mark the constant initialization convergence as a horizontal line
    """
    plt.figure(figsize=(7,5))  # Figure size
    plt.boxplot([convergence_gaussian, convergence_uniform], labels=["Gaussian", "Uniform"])  # Boxplot
    plt.axhline(y=convergence_constant, color='r', linestyle='--', label="Constant Init")  # Constant convergence
    plt.ylabel("Convergence Epoch")  # y-axis label
    plt.title("Convergence Times for Different Initialization Schemes")  # Title
    plt.legend()  # Show legend
    plt.grid(axis="y")  # Horizontal grid
    plt.show()  # Display plot

if __name__ == "__main__":
    points, labels = create_linear_dataset()  # Create linearly separable dataset
    convergence_gaussian = []  # Store convergence epochs
    for i in range(100):  # Train 100 models
        epoch = train_until_convergence(points, labels, lr=0.1, epochs=500, init_scheme="gaussian")
        convergence_gaussian.append(epoch)  # Store convergence time

    convergence_uniform = []  # Store convergence epochs
    for i in range(100):  # Train 100 models
        epoch = train_until_convergence(points, labels, lr=0.1, epochs=500, init_scheme="uniform")
        convergence_uniform.append(epoch)  # Store convergence time

    convergence_constant = train_until_convergence(points, labels, lr=0.1, epochs=500, init_scheme="constant")  # Only one model needed
    plot_convergence_box(convergence_gaussian, convergence_uniform, convergence_constant)  # Boxplot with constant marked
