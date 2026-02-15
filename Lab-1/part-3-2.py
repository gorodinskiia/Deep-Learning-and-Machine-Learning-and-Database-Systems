import torch
import math
import itertools
import matplotlib.pyplot as plt

# python3 part-3-2.py

def sigmoid(x): # sigmoid activation function
    return 1 / (1 + torch.exp(-x)) # sigmoid calculation

def sigmoid_derivative(x): # derivative of sigmoid function
    s = sigmoid(x) # compute sigmoid
    return s * (1 - s) # compute derivative

class TwoLayerNN: # two-layer neural network class
    def __init__(self, d_input, d_hidden=2, lr=0.1): # constructor
        self.lr = lr # learning rate
        self.w1 = torch.randn(d_hidden, d_input) / math.sqrt(d_input) # weights for first layer
        self.b1 = torch.zeros(d_hidden, 1) # biases for first layer
        self.w2 = torch.randn(d_hidden, 1) / math.sqrt(d_hidden) # weights for second layer
        self.b2 = torch.zeros(1, 1) # biases for second layer

    def forward(self, x): # forward pass
        self.p1 = self.w1 @ x.unsqueeze(1) + self.b1 # linear combination for first layer
        self.o1 = sigmoid(self.p1) # activation for first layer
        self.p2 = self.w2.T @ self.o1 + self.b2 # linear combination for second layer
        self.y_hat = sigmoid(self.p2) # activation for second layer
        return self.y_hat # return output

    def backward(self, x, y): # backward pass
        y = torch.tensor([[y]], dtype=torch.float32) # true label as tensor
        delta2 = (self.y_hat - y) * sigmoid_derivative(self.p2) # output layer error
        dw2 = self.o1 @ delta2.T # weight gradient for second layer
        db2 = delta2 # bias gradient for second layer
        delta1 = (self.w2 * delta2) * sigmoid_derivative(self.p1) # hidden layer error
        dw1 = delta1 @ x.unsqueeze(0) # weight gradient for first layer
        db1 = delta1 # bias gradient for first layer
        self.w2 -= self.lr * dw2 # update weights for second layer
        self.b2 -= self.lr * db2 # update biases for second layer
        self.w1 -= self.lr * dw1 # update weights for first layer
        self.b1 -= self.lr * db1 # update biases for first layer

    def predict(self, x): # prediction function
        return 1 if self.forward(x).item() >= 0.5 else 0 # return predicted label

def create_parity_dataset(d_input):
    vertices = list(itertools.product([-1.0, 1.0], repeat=d_input)) # Generate all vertices of d-dimensional hypercube
    points = torch.tensor(vertices, dtype=torch.float32) # Convert vertices to tensor
    labels = torch.tensor( # Create labels based on parity of -1.0 count
        [int(sum(1 for x in vertex if x == -1.0) % 2 == 1) for vertex in vertices], # 1 if odd number of -1.0s else 0
        dtype=torch.float32 # Set data type
    ).unsqueeze(1) # Reshape labels to be column vector
    return points, labels # Return points and labels

def train_model(points, labels, d_hidden=2, lr=0.1, epochs=1000): # Train the neural network model
    model = TwoLayerNN(d_input=points.shape[1], d_hidden=d_hidden, lr=lr) # Initialize model
    acc_history = [] # To store accuracy over epochs
    for epoch in range(epochs): # Training loop
        correct = 0 # initialize correct predictions counter
        for i in range(len(points)): # Iterate over each data point
            x = points[i] # Input features
            y = labels[i].item() # True label
            pred = model.predict(x) # Get prediction
            if pred == y: # If prediction is correct
                correct += 1 # Increment correct counter
            model.backward(x, y) # Backward pass to update weights and biases
        acc_history.append(correct / len(points)) # Calculate and store accuracy for the epoch
    return model, acc_history # Return trained model and accuracy history

def plot_per_dataset(results, d_inputs): # Plot accuracy vs epoch for each dataset
    for d in d_inputs: # Iterate over each input dimension
        plt.figure(figsize=(7,5)) # Create a new figure
        plt.plot(results[d]["input_dim_hidden"], label=f"Hidden={d}") # Plot accuracy for hidden=d
        plt.plot(results[d]["2_hidden"], label="Hidden=2") # Plot accuracy for hidden=2
        plt.title(f"Accuracy vs Epoch — {d}-D Parity") # Set title
        plt.xlabel("Epoch") # Set x-axis label
        plt.ylabel("Accuracy") # Set y-axis label
        plt.legend() # Show legend
        plt.grid(True) # Enable grid
        plt.show() # Display the plot

def plot_per_hidden(results, d_inputs, hidden_versions): # Plot accuracy vs epoch for each hidden version
    for hidden_version in hidden_versions: # Iterate over each hidden version
        plt.figure(figsize=(7,5)) # Create a new figure
        for d in d_inputs: # Iterate over each input dimension
            plt.plot(results[d][hidden_version], label=f"d={d}") # Plot accuracy for current hidden version
        plt.title(f"Accuracy vs Epoch — Hidden={hidden_version}") # Set title
        plt.xlabel("Epoch") # Set x-axis label
        plt.ylabel("Accuracy") # Set y-axis label
        plt.legend() # Show legend
        plt.grid(True) # Enable grid
        plt.show() # Display the plot

if __name__ == "__main__":

    d_inputs = [4, 6, 8] # Different input dimensions to test
    hidden_versions = ["input_dim_hidden", "2_hidden"] # Different hidden layer configurations
    results = {} # To store results
    lr = 0.1 # Learning rate
    epochs = 1000 # Number of training epochs

    for d in d_inputs: # Iterate over each input dimension
        print(f"\nGenerating {d}-D parity dataset...") # Log dataset generation
        points, labels = create_parity_dataset(d) # Create parity dataset
        results[d] = {} # Initialize results dictionary for current dimension
        print(f"Training model with hidden={d} neurons...") # Log training start
        _, acc_input_hidden = train_model(points, labels, d_hidden=d, lr=lr, epochs=epochs) # Train model with hidden=d
        results[d]["input_dim_hidden"] = acc_input_hidden # Store accuracy history
        print(f"Training model with hidden=2 neurons...") # Log training start
        _, acc_2_hidden = train_model(points, labels, d_hidden=2, lr=lr, epochs=epochs) # Train model with hidden=2
        results[d]["2_hidden"] = acc_2_hidden # Store accuracy history

    print("\nPlotting accuracy vs epoch per dataset...") # Log plotting start
    plot_per_dataset(results, d_inputs) # Plot accuracy per dataset
    print("\nPlotting accuracy vs epoch per hidden version...") # Log plotting start
    plot_per_hidden(results, d_inputs, hidden_versions) # Plot accuracy per hidden version
    print("\nExperiment Completed!") # Log experiment completion
