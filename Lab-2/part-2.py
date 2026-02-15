import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Configuration
# -------------------------------
DEVICE = "cpu"
EPOCHS = 5
LEARNING_RATE = 0.01
BATCH_SIZES = [1, 16, 32]
MOMENTUM_VALUES = [0.0, 0.9]
HIDDEN_DIM = 256

# -------------------------------
# Load Dataset
# -------------------------------
print("Loading dataset...")
dataset = load_dataset("stevengubkin/mathoverflow_text_arxiv_labels")

texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

num_classes = len(set(labels))
print(f"Classes: {num_classes}")

# -------------------------------
# Text Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# -------------------------------
# Normalization
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Train / Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

INPUT_DIM = X_train.shape[1]

# -------------------------------
# Activation Functions
# -------------------------------
def leaky_relu(x):
    return torch.maximum(x, 0.1 * x)

def tanh(x):
    return torch.tanh(x)

# -------------------------------
# Models
# -------------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class DeepNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, num_classes)
        ])
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return F.softmax(self.layers[-1](x), dim=1)

# -------------------------------
# Loss Function
# -------------------------------
def cross_entropy(pred, target):
    return -torch.log(pred[range(len(target)), target]).mean()

# -------------------------------
# Training Loop (SGD / Mini-batch / Momentum)
# -------------------------------
def train_model(model, batch_size=1, momentum=0.0):
    model.train()
    velocity = [torch.zeros_like(p) for p in model.parameters()]

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_train[idx]
            y_batch = y_train[idx]

            preds = model(x_batch)
            loss = cross_entropy(preds, y_batch)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for j, p in enumerate(model.parameters()):
                    velocity[j] = momentum * velocity[j] + p.grad
                    p -= LEARNING_RATE * velocity[j]

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        predicted = torch.argmax(preds, dim=1)
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy

# -------------------------------
# Experiments
# -------------------------------
print("\n=== Dataset Difficulty Check ===")
shallow = TwoLayerNet(torch.sigmoid)
train_model(shallow, batch_size=1)
print("Shallow Accuracy:", evaluate(shallow))

print("\n=== Baseline Deep Model (Sigmoid) ===")
baseline = DeepNet(torch.sigmoid)
train_model(baseline, batch_size=1)
print("Baseline Accuracy:", evaluate(baseline))

print("\n=== Activation Function Experiments ===")
for name, act in [("Leaky ReLU", leaky_relu), ("Tanh", tanh)]:
    print(f"\n{name}")
    model = DeepNet(act)
    train_model(model, batch_size=1)
    print("Accuracy:", evaluate(model))

print("\n=== Mini-batch SGD Experiments ===")
best_activation = leaky_relu
for b in BATCH_SIZES:
    print(f"\nBatch Size: {b}")
    model = DeepNet(best_activation)
    train_model(model, batch_size=b)
    print("Accuracy:", evaluate(model))

print("\n=== Momentum Experiments ===")
for alpha in MOMENTUM_VALUES:
    print(f"\nMomentum α = {alpha}")
    model = DeepNet(best_activation)
    train_model(model, batch_size=32, momentum=alpha)
    print("Accuracy:", evaluate(model))
