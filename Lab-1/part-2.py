import torch
import math
import matplotlib.pyplot as plt

# python3 part-2.py

def create_linearly_separable_dataset(): # We define a function to create the linearly separable dataset
    S = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]) # This is the S set we were given turned into a 1D tensor
    x1, x2 = torch.meshgrid(S, S, indexing="ij") # This created a grid of all possible combinations of points in S
    points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1) # This reshaped the grid into a list of 2D points
    labels = (points[:, 1] > points[:, 0]).int().unsqueeze(1) # This created the labels based on the condition y > x
    return points, labels # We return the points and labels

def create_xor_dataset(): # We define a function to create the XOR dataset
    points = torch.tensor([
        [ 1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0, -1.0],
        [-1.0,  1.0]
    ]) # The input points for the XOR problem
    labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1) # The corresponding labels for the XOR problem
    return points, labels # We return the points and labels

def sigmoid(x): # Sigmoid activation function
    return 1 / (1 + torch.exp(-x)) # Sigmoid formula

def sigmoid_derivative(x): # Derivative of sigmoid function
    s = sigmoid(x) # Compute sigmoid
    return s * (1 - s) # Derivative formula

class TwoLayerNN:
    
    def __init__(self, dimensions_input, dimensions_hidden=2, learning_rate=0.1):
        self.dimensions_input = dimensions_input # Input feature dimensions
        self.dimensions_hidden = dimensions_hidden # Hidden layer dimensions
        self.learning_rate = learning_rate # Learning rate

        # Initialize weights and biases
        self.w1 = torch.randn(dimensions_hidden, dimensions_input) / math.sqrt(dimensions_input) # Gaussian initialization
        self.b1 = torch.zeros(dimensions_hidden, 1) # column vector
        self.w2 = torch.randn(dimensions_hidden, 1) / math.sqrt(dimensions_hidden) # column vector
        self.b2 = torch.zeros(1, 1) # column vector
    
    def forward(self, x):
        self.p1 = self.w1 @ x.unsqueeze(1) + self.b1   # dhidden x 1
        self.o1 = sigmoid(self.p1)                     # dhidden x 1
        self.p2 = (self.w2.T @ self.o1) + self.b2     # 1 x 1
        self.y_hat = sigmoid(self.p2)                  # 1 x 1
        return self.y_hat
    
    def backward(self, x, y):
        y = torch.tensor([[y]], dtype=torch.float32)  # 1x1 tensor

        # Output layer
        delta_y = self.y_hat - y                       # 1x1
        delta_p2 = delta_y * sigmoid_derivative(self.p2)  # 1x1
        delta_w2 = self.o1 @ delta_p2.T               # dhidden x 1
        delta_b2 = delta_p2                            # 1x1

        # Hidden layer
        delta_o1 = self.w2 * delta_p2                 # dhidden x 1
        delta_p1 = delta_o1 * sigmoid_derivative(self.p1)  # dhidden x 1
        delta_w1 = delta_p1 @ x.unsqueeze(0)          # dhidden x d_input
        delta_b1 = delta_p1                            # dhidden x 1

        # Update parameters (FIXED: use self.learning_rate)
        self.w2 -= self.learning_rate * delta_w2
        self.b2 -= self.learning_rate * delta_b2
        self.w1 -= self.learning_rate * delta_w1
        self.b1 -= self.learning_rate * delta_b1
    
    def predict(self, x):
        y_hat = self.forward(x) # forward pass
        return 1 if y_hat.item() >= 0.5 else 0  # convert 1x1 tensor to scalar

def train_two_layer(points, labels, lr=0.1, epochs=1000):
    model = TwoLayerNN(dimensions_input=points.shape[1], learning_rate=lr) # FIXED argument name
    accuracy_history = []

    for epoch in range(epochs):
        correct = 0
        for i in range(len(points)):
            x = points[i].float()
            y = labels[i].item()
            
            y_hat = model.forward(x)
            pred = 1 if y_hat.item() >= 0.5 else 0
            if pred == y:
                correct += 1
            model.backward(x, y)
        
        accuracy = correct / len(points)
        accuracy_history.append(accuracy)

    return model, accuracy_history

def count_perfect_accuracy(models):
    perfect_count = sum(1 for _, acc_hist in models if acc_hist[-1] == 1.0) # Count models with perfect accuracy
    return perfect_count # Return the count of models that achieved perfect accuracy

def select_best_model(models): # Select the model that achieved perfect accuracy the fastest
    best_epochs = float('inf') # Initialize best epochs to infinity
    best_model = None # Initialize best model
    best_acc_hist = None # Initialize best accuracy history
    # First try: perfect accuracy
    for model, acc_hist in models: # Iterate through all models
        try:
            first_perfect_epoch = next(i for i, acc in enumerate(acc_hist) if acc == 1.0)
        except StopIteration:
            continue
        if first_perfect_epoch < best_epochs:
            best_epochs = first_perfect_epoch
            best_model = model
            best_acc_hist = acc_hist
    # Fallback: pick model with highest final accuracy
    if best_model is None:
        best_model, best_acc_hist = max(
            models, key=lambda item: item[1][-1]
        )
    return best_model, best_acc_hist


def plot_accuracy(acc_hist, dataset_name): # Plot accuracy over epochs
    plt.figure(figsize=(7,5)) # Create a figure
    plt.plot(range(1, len(acc_hist)+1), acc_hist, color='blue') # Plot accuracy
    plt.xlabel('Epoch') # X-axis label
    plt.ylabel('Accuracy') # Y-axis label
    plt.title(f'Accuracy vs Epoch for {dataset_name}') # Title
    plt.ylim(0, 1.05) # Y-axis limits
    plt.grid() # Grid
    plt.show() # Show plot

def plot_decision_boundary_grid(model, dataset_name): # Plot decision boundary using grid sampling
    grid_size = 20  # 20x20 -> 400 points
    x_vals = torch.linspace(-1, 1, grid_size) # X values
    y_vals = torch.linspace(-1, 1, grid_size) # Y values
    xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij') # Create meshgrid
    grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1) # Reshape to list of 2D points

    preds = [] # Store predictions
    for i in range(len(grid_points)): # Iterate through grid points
        x = grid_points[i].float() # Current point
        pred = model.predict(x) # Predict class
        preds.append(pred) # Append prediction

    preds = torch.tensor(preds) # Convert to tensor

    plt.figure(figsize=(6,6)) # Create figure
    plt.scatter(grid_points[:,0], grid_points[:,1], c=preds, cmap='bwr', s=40) # Scatter plot
    plt.xlabel('x1') # X-axis label
    plt.ylabel('x2') # Y-axis label
    plt.title(f'{dataset_name} Decision Boundary (Grid Sampling)') # Title
    plt.grid() # Grid
    plt.show() # Show plot

if __name__ == "__main__":
    points_ls, labels_ls = create_linearly_separable_dataset()
    models_ls = [train_two_layer(points_ls, labels_ls, lr=0.1, epochs=1000) for _ in range(10)]
    perfect_ls = count_perfect_accuracy(models_ls)
    print(f"Linearly separable: {perfect_ls}/10 runs achieved perfect accuracy")
    best_model_ls, best_acc_ls = select_best_model(models_ls)
    plot_accuracy(best_acc_ls, "Linearly Separable")
    plot_decision_boundary_grid(best_model_ls, "Linearly Separable")

    points_xor, labels_xor = create_xor_dataset() # Create XOR dataset
    models_xor = [train_two_layer(points_xor, labels_xor, lr=0.1, epochs=1000) for _ in range(10)] # Train 10 models
    perfect_xor = count_perfect_accuracy(models_xor) # Count perfect accuracy models
    print(f"XOR: {perfect_xor}/10 runs achieved perfect accuracy") # Print result
    best_model_xor, best_acc_xor = select_best_model(models_xor) # Select best model
    plot_accuracy(best_acc_xor, "XOR") # Plot accuracy
    plot_decision_boundary_grid(best_model_xor, "XOR") # Plot decision boundary
