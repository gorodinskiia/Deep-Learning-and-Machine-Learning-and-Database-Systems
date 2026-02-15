import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# python3 part-1.py

def create_linearly_separable_dataset(): # We define a function to create the linearly separable dataset
    S = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]) # This is the S set we were given turned into a 1D tensor
    x1, x2 = torch.meshgrid(S, S, indexing="ij") # This created a grid of all possible combinations of points in S
    points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1) # This reshaped the grid into a list of 2D points
    labels = (points[:, 1] > points[:, 0]).int().unsqueeze(1) # This created the labels based on the condition y > x
    return points, labels # We return the points and labels

def create_xor_dataset():
    points = torch.tensor([
        [ 1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0, -1.0],
        [-1.0,  1.0]
    ]) # The input points for the XOR problem
    labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1) # The corresponding labels for the XOR problem
    return points, labels # We return the points and labels

class Perceptron:
    
    def __init__ (self, input_dimensions, learning_rate): # Constructor to initialize weights and bias
        self.input_dimensions = input_dimensions # Number of input features
        self.learning_rate = learning_rate # Learning rate for weight updates
        self.weights = torch.randn(input_dimensions) / math.sqrt(input_dimensions) # Weights initialization
        self.bias = torch.tensor(0.0) # Bias initialization
        
    def forward(self, x): # Forward pass to compute the output
        z = torch.dot(self.weights, x) + self.bias # Linear combination of inputs and weights
        y_hat = 1 if z > 0 else 0 # Step activation function
        return y_hat, z # Return predicted label and linear output
    
    def backward(self, x, y, y_hat): # Backward pass to update weights and bias
        error = y - y_hat # Compute the error
        self.weights += self.learning_rate * error * x # Update weights
        self.bias += self.learning_rate * error # Update bias

def train_perceptron(points, labels, learning_rate, epochs=100): # Train the perceptron on the given dataset
    model = Perceptron(input_dimensions=points.shape[1], learning_rate=learning_rate) # Initialize the perceptron model
    accuracy_history = [] # To store accuracy over epochs
    angle_history = [] # To store decision boundary angle over epochs

    for epoch in range(epochs): # Training loop for the specified number of epochs
        correct = 0 # Counter for correct predictions

        for i in range(len(points)): # Iterate over each data point
            x = points[i] # Input features
            y = labels[i].item() # True label
            y_hat, _ = model.forward(x) # Forward pass to get prediction
            if y_hat == y: # If prediction is correct
                correct += 1 # Increment correct counter
            model.backward(x, y, y_hat) # Backward pass to update weights and bias

        accuracy = correct / len(points) # Calculate accuracy for the epoch
        accuracy_history.append(accuracy) # Store accuracy
        w1, w2 = model.w[0].item(), model.w[1].item() # Extract weights
        theta = math.atan2(w2, w1) + math.pi / 2 # Calculate decision boundary angle
        angle_history.append(theta) # Store angle

    return model, accuracy_history, angle_history # Return trained model and training metrics

def plot_training_metrics(acc_history, angle_history, dataset_name): # Plot accuracy and decision boundary angle over epochs
    epochs = list(range(1, len(acc_history)+1)) # Epoch numbers
    fig, ax1 = plt.subplots(figsize=(7,5)) # Create a figure and axis
    color = 'tab:blue' # Color for accuracy plot
    ax1.set_xlabel('Epoch') # X-axis label
    ax1.set_ylabel('Accuracy', color=color) # Y-axis label for accuracy
    ax1.plot(epochs, acc_history, color=color, label='Accuracy') # Plot accuracy
    ax1.tick_params(axis='y', labelcolor=color) # Set y-axis tick color
    ax1.set_ylim(0, 1.05) # Set y-axis limits for accuracy
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red' # Color for angle plot
    ax2.set_ylabel('Decision Boundary Angle (radians)', color=color) # Y-axis label for angle
    ax2.plot(epochs, angle_history, color=color, linestyle='--', label='Boundary Angle') # Plot angle
    ax2.tick_params(axis='y', labelcolor=color) # Set y-axis tick color
    fig.suptitle(f'Training Metrics for {dataset_name} Dataset') # Title for the plot
    fig.tight_layout() # Adjust layout
    plt.show() # Show the plot


def plot_decision_boundary(points, labels, model, dataset_name): # Plot the dataset with the learned decision boundary
    plt.figure(figsize=(6,6)) # Create a figure
    plt.scatter(points[:, 0], points[:, 1], c=labels.squeeze(), cmap='bwr', s=100) # Scatter plot of data points
    # Calculate decision boundary: w1*x + w2*y + b = 0 => y = -(w1/w2)x - b/w2
    w1, w2 = model.w[0].item(), model.w[1].item() # Extract weights
    b = model.b.item() # Extract bias
    x_vals = np.array([points[:,0].min()-0.5, points[:,0].max()+0.5]) # X values for the line
    y_vals = -(w1/w2) * x_vals - b/w2 # Corresponding Y values for the line
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary') # Plot decision boundary
    plt.xlabel('x1') # X-axis label
    plt.ylabel('x2') # Y-axis label
    plt.title(f'{dataset_name} Dataset with Learned Decision Boundary') # Title for the plot
    plt.grid() # Show grid
    plt.legend() # Show legend
    plt.show() # Show the plot


if __name__ == "__main__":
    # Linearly separable dataset
    points_ls, labels_ls = create_linearly_separable_dataset()
    model_ls, acc_ls, angle_ls = train_perceptron(points_ls, labels_ls, learning_rate=0.001, epochs=100)
    print("Linearly Separable Dataset")
    print("Final accuracy:", acc_ls[-1])
    print("Final decision boundary angle:", angle_ls[-1])
    plot_training_metrics(acc_ls, angle_ls, "Linearly Separable")
    plot_decision_boundary(points_ls, labels_ls, model_ls, "Linearly Separable")

    # XOR dataset
    points_xor, labels_xor = create_xor_dataset()
    model_xor, acc_xor, angle_xor = train_perceptron(points_xor, labels_xor, learning_rate=0.1, epochs=100)
    print("XOR Dataset")
    print("Final accuracy:", acc_xor[-1])
    print("Final decision boundary angle:", angle_xor[-1])
    plot_training_metrics(acc_xor, angle_xor, "XOR")
    plot_decision_boundary(points_xor, labels_xor, model_xor, "XOR")



    
        