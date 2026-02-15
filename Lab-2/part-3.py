import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cpu"
EPOCHS = 5
LEARNING_RATE = 0.01
HIDDEN_DIM = 256
BATCH_SIZES = [1, 16, 32]
MOMENTUM_VALUES = [0.0, 0.9]

# ============================================================
# LOAD DATASET
# ============================================================
print("Loading dataset...")
dataset = load_dataset("stevengubkin/mathoverflow_text_arxiv_labels")

texts = dataset["train"]["text"]
labels = dataset["train"]["label"]
num_classes = len(set(labels))

# ============================================================
# TEXT VECTORIZATION
# ============================================================
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# ============================================================
# NORMALIZATION
# ============================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================================================
# TRAIN / TEST SPLIT (70/30)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

INPUT_DIM = X_train.shape[1]

# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================
def leaky_relu(x):
    return torch.maximum(x, 0.1 * x)

def tanh(x):
    return torch.tanh(x)

# ============================================================
# LOSS FUNCTION
# ============================================================
def cross_entropy(pred, target):
    return -torch.log(pred[range(len(target)), target]).mean()

# ============================================================
# TRAINING + EVALUATION UTILITIES
# ============================================================
def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        predicted = torch.argmax(preds, dim=1)
        return (predicted == y_test).float().mean().item()

def average_gradient_l1(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.abs().sum().item()
            count += 1
    return total / count if count > 0 else 0.0

def train_model(model, batch_size=1, momentum=0.0, track_grad=False):
    model.train()
    velocity = [torch.zeros_like(p) for p in model.parameters()]
    grad_norms = []

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

            if track_grad and epoch == 0:
                grad_norms.append(average_gradient_l1(model))

            with torch.no_grad():
                for j, p in enumerate(model.parameters()):
                    velocity[j] = momentum * velocity[j] + p.grad
                    p -= LEARNING_RATE * velocity[j]

    return np.mean(grad_norms) if track_grad else None

# ============================================================
# MODELS
# ============================================================
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
        self.activation = activation
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, num_classes)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return F.softmax(self.layers[-1](x), dim=1)

class ExtendedDeepNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM)] +
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(10)] +
            [nn.Linear(HIDDEN_DIM, num_classes)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return F.softmax(self.layers[-1](x), dim=1)

class SkipNetA(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM)] +
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(10)] +
            [nn.Linear(HIDDEN_DIM, num_classes)]
        )

    def forward(self, x):
        x = self.activation(self.layers[0](x))
        a = self.activation(self.layers[1](x))
        b = self.activation(self.layers[2](a))
        x = a + b
        c = self.activation(self.layers[3](x))
        d = self.activation(self.layers[4](c))
        x = c + d
        e = self.activation(self.layers[5](x))
        f = self.activation(self.layers[6](e))
        x = e + f
        for layer in self.layers[7:-1]:
            x = self.activation(layer(x))
        return F.softmax(self.layers[-1](x), dim=1)

class SkipNetB(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM)] +
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(10)] +
            [nn.Linear(HIDDEN_DIM, num_classes)]
        )

    def forward(self, x):
        x = self.activation(self.layers[0](x))
        a = self.activation(self.layers[1](x))
        b = self.activation(self.layers[2](a))
        c = self.activation(self.layers[3](b))
        x = a + c
        d = self.activation(self.layers[4](x))
        e = self.activation(self.layers[5](d))
        f = self.activation(self.layers[6](e))
        g = self.activation(self.layers[7](f))
        x = d + g
        h = self.activation(self.layers[8](x))
        i = self.activation(self.layers[9](h))
        j = self.activation(self.layers[10](i))
        x = h + j
        return F.softmax(self.layers[-1](x), dim=1)

# ============================================================
# EXPERIMENTS
# ============================================================
print("\n--- Dataset Difficulty Check ---")
shallow = TwoLayerNet(torch.sigmoid)
train_model(shallow, batch_size=1)
print("Shallow Accuracy:", evaluate(shallow))

print("\n--- Baseline Deep Model (Sigmoid) ---")
baseline = DeepNet(torch.sigmoid)
train_model(baseline, batch_size=1)
print("Baseline Accuracy:", evaluate(baseline))

print("\n--- Activation Function Experiments ---")
for name, act in [("Leaky ReLU", leaky_relu), ("Tanh", tanh)]:
    model = DeepNet(act)
    train_model(model, batch_size=1)
    print(f"{name} Accuracy:", evaluate(model))

BEST_ACTIVATION = leaky_relu
BEST_BATCH_SIZE = 32
BEST_MOMENTUM = 0.9

print("\n--- Extended Model (No Skip Connections) ---")
ext = ExtendedDeepNet(BEST_ACTIVATION)
grad_ext = train_model(ext, BEST_BATCH_SIZE, BEST_MOMENTUM, True)
print("Accuracy:", evaluate(ext))
print("Avg L1 Gradient:", grad_ext)

print("\n--- Skip Connection Model A ---")
skip_a = SkipNetA(BEST_ACTIVATION)
grad_a = train_model(skip_a, BEST_BATCH_SIZE, BEST_MOMENTUM, True)
print("Accuracy:", evaluate(skip_a))
print("Avg L1 Gradient:", grad_a)

print("\n--- Skip Connection Model B ---")
skip_b = SkipNetB(BEST_ACTIVATION)
grad_b = train_model(skip_b, BEST_BATCH_SIZE, BEST_MOMENTUM, True)
print("Accuracy:", evaluate(skip_b))
print("Avg L1 Gradient:", grad_b)
