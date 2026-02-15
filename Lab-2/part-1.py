import torch
import numpy as np
from datasets import load_dataset
from collections import Counter
import random

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("stevengubkin/mathoverflow_text_arxiv_labels")["train"]

label_cols = [c for c in dataset.column_names if c != "Title_Body"]

texts = []
labels = []

for row in dataset:
    active = [i for i, c in enumerate(label_cols) if row[c] == 1]
    if len(active) == 1:
        texts.append(row["Title_Body"])
        labels.append(active[0])

labels = np.array(labels)
num_classes = len(set(labels))

print(f"Samples after filtering: {len(texts)}")
print(f"Number of classes: {num_classes}")

# -----------------------------
# Text Preprocessing (BoW)
# -----------------------------
def tokenize(text):
    return text.lower().split()

vocab_counter = Counter()
for t in texts:
    vocab_counter.update(tokenize(t))

VOCAB_SIZE = 5000
vocab = {w: i for i, (w, _) in enumerate(vocab_counter.most_common(VOCAB_SIZE))}

def vectorize(text):
    vec = np.zeros(VOCAB_SIZE)
    for w in tokenize(text):
        if w in vocab:
            vec[vocab[w]] += 1
    return vec

X = np.vstack([vectorize(t) for t in texts])
y = labels

# -----------------------------
# Normalization
# -----------------------------
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X = (X - mean) / std

# -----------------------------
# Train/Test Split (70/30)
# -----------------------------
idx = np.random.permutation(len(X))
split = int(0.7 * len(X))

train_idx = idx[:split]
test_idx = idx[split:]

X_train = torch.tensor(X[train_idx], dtype=torch.float32)
y_train = torch.tensor(y[train_idx], dtype=torch.long)

X_test = torch.tensor(X[test_idx], dtype=torch.float32)
y_test = torch.tensor(y[test_idx], dtype=torch.long)

# -----------------------------
# Activations (From Scratch)
# -----------------------------
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def softmax(x):
    e = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return e / e.sum(dim=1, keepdim=True)

# -----------------------------
# Loss Function
# -----------------------------
def cross_entropy(pred, target):
    return -torch.log(pred[0, target] + 1e-8)

# -----------------------------
# Two-Layer Network
# -----------------------------
class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim, requires_grad=True) * 0.01
        self.b1 = torch.zeros(hidden_dim, requires_grad=True)
        self.W2 = torch.randn(hidden_dim, output_dim, requires_grad=True) * 0.01
        self.b2 = torch.zeros(output_dim, requires_grad=True)

    def forward(self, x):
        h = sigmoid(x @ self.W1 + self.b1)
        return softmax(h @ self.W2 + self.b2)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

# -----------------------------
# Deep Baseline Network (5 layers)
# -----------------------------
class DeepNet:
    def __init__(self, dims):
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            W = torch.randn(dims[i], dims[i+1], requires_grad=True) * 0.01
            b = torch.zeros(dims[i+1], requires_grad=True)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = sigmoid(x @ W + b)
        return softmax(x @ self.weights[-1] + self.biases[-1])

    def params(self):
        return self.weights + self.biases

# -----------------------------
# Training Loop (SGD, batch=1)
# -----------------------------
def train(model, X, y, lr=0.01, epochs=5):
    for epoch in range(epochs):
        correct = 0
        for i in range(len(X)):
            x = X[i].unsqueeze(0)
            target = y[i]

            pred = model.forward(x)
            loss = cross_entropy(pred, target)

            for p in model.params():
                p.grad = None

            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad

            if pred.argmax(dim=1).item() == target.item():
                correct += 1

        acc = correct / len(X)
        print(f"Epoch {epoch+1}: Train Accuracy = {acc:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X, y):
    correct = 0
    with torch.no_grad():
        for i in range(len(X)):
            pred = model.forward(X[i].unsqueeze(0))
            if pred.argmax(dim=1).item() == y[i].item():
                correct += 1
    return correct / len(X)

# -----------------------------
# Run Experiments
# -----------------------------
print("\nTraining Two-Layer Network")
two_layer = TwoLayerNet(VOCAB_SIZE, 128, num_classes)
train(two_layer, X_train, y_train, lr=0.01, epochs=5)
print("Two-Layer Test Accuracy:", evaluate(two_layer, X_test, y_test))

print("\nTraining Deep Network")
deep_dims = [VOCAB_SIZE, 512, 256, 128, 64, num_classes]
deep_net = DeepNet(deep_dims)
train(deep_net, X_train, y_train, lr=0.01, epochs=5)
print("Deep Network Test Accuracy:", evaluate(deep_net, X_test, y_test))
