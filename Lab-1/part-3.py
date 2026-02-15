import torch  # PyTorch library for tensors and math operations
import math   # Standard math library for sqrt, etc.
import matplotlib.pyplot as plt  # Library for plotting graphs
import random  # Python random library (not used in this code)

def create_xor_dataset():
    points = torch.tensor([  # Define the four XOR points in 2D
        [ 1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0, -1.0],
        [-1.0,  1.0]
    ])
    labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1)  # XOR labels, unsqueeze to make column vector
    return points, labels  # Return points and labels


def create_noisy_xor_dataset(r):
    points, labels = create_xor_dataset()  # Get the base XOR dataset
    points = points.repeat(10, 1)  # Repeat each point 10 times (40 samples total)
    labels = labels.repeat(10, 1)  # Repeat labels accordingly
    noise = torch.randn_like(points) * r  # Generate Gaussian noise scaled by r
    points_noisy = points + noise  # Add noise to the points
    return points_noisy, labels  # Return noisy points and labels

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))  # Sigmoid activation function

def sigmoid_derivative(x):
    s = sigmoid(x)  # Compute sigmoid of x
    return s * (1 - s)  # Derivative of sigmoid for backpropagation

class TwoLayerNN:
    def __init__(self, d_input, d_hidden=2, lr=0.1):
        self.lr = lr  # Learning rate
        self.w1 = torch.randn(d_hidden, d_input) / math.sqrt(d_input)  # Input-to-hidden weights, scaled
        self.b1 = torch.zeros(d_hidden, 1)  # Hidden layer biases initialized to zeros
        self.w2 = torch.randn(d_hidden, 1) / math.sqrt(d_hidden)  # Hidden-to-output weights, scaled
        self.b2 = torch.zeros(1, 1)  # Output bias initialized to zero

    def forward(self, x):
        self.p1 = self.w1 @ x.unsqueeze(1) + self.b1  # Linear combination for hidden layer
        self.o1 = sigmoid(self.p1)  # Apply sigmoid activation to hidden layer
        self.p2 = self.w2.T @ self.o1 + self.b2  # Linear combination for output layer
        self.y_hat = sigmoid(self.p2)  # Apply sigmoid activation to output
        return self.y_hat  # Return predicted output

    def backward(self, x, y):
        y = torch.tensor([[y]], dtype=torch.float32)  # Convert target to tensor with shape (1,1)
        delta2 = (self.y_hat - y) * sigmoid_derivative(self.p2)  # Output layer error term
        dw2 = self.o1 @ delta2.T  # Gradient for output weights
        db2 = delta2  # Gradient for output bias
        delta1 = (self.w2 * delta2) * sigmoid_derivative(self.p1)  # Hidden layer error term
        dw1 = delta1 @ x.unsqueeze(0)  # Gradient for hidden weights
        db1 = delta1  # Gradient for hidden biases
        self.w2 -= self.lr * dw2  # Update output weights
        self.b2 -= self.lr * db2  # Update output bias
        self.w1 -= self.lr * dw1  # Update hidden weights
        self.b1 -= self.lr * db1  # Update hidden biases

    def predict(self, x):
        return 1 if self.forward(x).item() >= 0.5 else 0  # Return binary prediction

def train_model(points, labels, lr=0.1, epochs=1000):
    model = TwoLayerNN(d_input=2, lr=lr)  # Initialize two-layer network
    acc_history = []  # List to store accuracy per epoch
    for epoch in range(epochs):
        correct = 0  # Reset correct count at start of epoch
        for i in range(len(points)):
            x = points[i]  # Get input sample
            y = labels[i].item()  # Get target label
            pred = model.predict(x)  # Make prediction
            if pred == y:  # Check if prediction is correct
                correct += 1  # Increment correct counter

            model.backward(x, y)  # Update weights via backpropagation

        if epoch % 100 == 0:  # Print progress every 100 epochs
            print(f"Epoch {epoch}/{epochs} — Current accuracy: {correct/len(points):.2f}")

        acc_history.append(correct / len(points))  # Store epoch accuracy

    return model, acc_history  # Return trained model and accuracy history

def plot_noisy_xor(points, labels, r):
    plt.figure(figsize=(6,6))  # Create figure with size 6x6
    plt.scatter(
        points[:,0],  # x-coordinates
        points[:,1],  # y-coordinates
        c=labels.squeeze(),  # Color by label
        cmap="bwr",  # Blue-red colormap
        s=40,  # Marker size
        alpha=0.7  # Transparency
    )
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)  # Horizontal axis line
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)  # Vertical axis line
    plt.xlabel("x1")  # Label x-axis
    plt.ylabel("x2")  # Label y-axis
    plt.title(f"Noisy XOR Dataset (r = {r})")  # Plot title
    plt.grid()  # Show grid
    plt.show()  # Display plot


def plot_boxplot(final_accuracies):
    plt.figure(figsize=(7,5))  # Create figure for boxplot
    plt.boxplot(
        [final_accuracies[r] for r in [0.25, 0.5, 0.75]],  # Data to plot
        labels=["r = 0.25", "r = 0.5", "r = 0.75"]  # Labels for each box
    )
    plt.ylabel("Final Accuracy")  # y-axis label
    plt.title("Final Accuracy on Noisy XOR Datasets")  # Plot title
    plt.grid(axis="y")  # Show horizontal grid lines
    plt.show()  # Display boxplot


def plot_decision_boundary(model, title):
    grid = torch.linspace(-2, 2, 200)  # Create grid for plotting decision boundary
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")  # Create 2D meshgrid
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # Flatten grid to list of points
    preds = torch.tensor([model.predict(p) for p in grid_points])  # Predict class for each grid point
    plt.figure(figsize=(6,6))  # Create figure
    plt.scatter(
        grid_points[:,0],  # x-coordinates
        grid_points[:,1],  # y-coordinates
        c=preds,  # Color by predicted class
        cmap="bwr",  # Blue-red colormap
        alpha=0.25,  # Transparency for background
        s=10  # Marker size
    )
    plt.xlabel("x1")  # x-axis label
    plt.ylabel("x2")  # y-axis label
    plt.title(title)  # Plot title
    plt.grid()  # Show grid
    plt.show()  # Display plot

if __name__ == "__main__":
    
    print("SCRIPT STARTED")  # Print when script starts
    noise_levels = [0.25, 0.5, 0.75]  # List of noise levels to test
    final_accuracies = {}  # Dictionary to store accuracies of all models
    best_models = {}  # Dictionary to store best-performing model per noise level

    for r in noise_levels:  # Loop over noise levels
        points, labels = create_noisy_xor_dataset(r)  # Generate noisy dataset
        plot_noisy_xor(points, labels, r)  # Plot the noisy dataset
        accuracies = []  # Store accuracies of all trained models
        best_acc = 0  # Track best accuracy
        best_model = None  # Track best model
        for i in range(100):  # Train 100 models
            model, acc_hist = train_model(points, labels)  # Train model
            final_acc = acc_hist[-1]  # Get final accuracy
            accuracies.append(final_acc)  # Add to list
            if final_acc > best_acc:  # Check if this model is best so far
                best_acc = final_acc
                best_model = model

            print(f"Model {i+1}/100 trained, final accuracy: {final_acc:.2f}, best so far: {best_acc:.2f}")  # Print progress

        final_accuracies[r] = accuracies  # Save all accuracies for this noise level
        best_models[r] = best_model  # Save best model
    plot_boxplot(final_accuracies)  # Plot boxplot of final accuracies
    for r in noise_levels:  # Plot decision boundary of best models
        plot_decision_boundary(
            best_models[r],
            f"Decision Boundary — Noisy XOR (r = {r})"
        )
