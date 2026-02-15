import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
EPOCHS = 5
LR = 0.01
HIDDEN_DIM = 256
BATCH_SIZES = [1, 16, 32]
MOMENTUM_VALUES = [0.0, 0.9]
WEIGHT_DECAY_VALUES = [0.0, 0.0001, 0.001]

# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================
print("Loading dataset...")
dataset = load_dataset("stevengubkin/mathoverflow_text_arxiv_labels")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]
num_classes = len(set(labels))

vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

INPUT_DIM = X_train.shape[1]

# ============================================================
# ACTIVATIONS
# ============================================================
def leaky_relu(x):
    return torch.maximum(x, 0.1 * x)

def tanh(x):
    return torch.tanh(x)

# ============================================================
# LOSS
# ============================================================
def cross_entropy(pred, target):
    return -torch.log(pred[range(len(target)), target]).mean()

# ============================================================
# UTILITIES
# ============================================================
def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        return (preds.argmax(dim=1) == y_test).float().mean().item()

def avg_grad_l1(model):
    total, count = 0.0, 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.abs().sum().item()
            count += 1
    return total / count if count > 0 else 0.0

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train(model, batch_size=1, momentum=0.0, weight_decay=0.0, track_grad=False):
    model.train()
    velocity = [torch.zeros_like(p) for p in model.parameters()]
    grad_log = []

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            x, y = X_train[idx], y_train[idx]

            preds = model(x)
            loss = cross_entropy(preds, y)

            model.zero_grad()
            loss.backward()

            if track_grad and epoch == 0:
                grad_log.append(avg_grad_l1(model))

            with torch.no_grad():
                for j, p in enumerate(model.parameters()):
                    velocity[j] = (
                        momentum * velocity[j]
                        + p.grad
                        + weight_decay * p
                    )
                    p -= LR * velocity[j]

    return np.mean(grad_log) if track_grad else None

# ============================================================
# MODELS
# ============================================================
class TwoLayerNet(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
        self.act = act

    def forward(self, x):
        return F.softmax(self.fc2(self.act(self.fc1(x))), dim=1)

class DeepNet(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, num_classes)
        ])

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.act(l(x))
        return F.softmax(self.layers[-1](x), dim=1)

class ExtendedNet(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM)] +
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(10)] +
            [nn.Linear(HIDDEN_DIM, num_classes)]
        )

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.act(l(x))
        return F.softmax(self.layers[-1](x), dim=1)

class SkipNetA(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM)] +
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(10)] +
            [nn.Linear(HIDDEN_DIM, num_classes)]
        )

    def forward(self, x):
        x = self.act(self.layers[0](x))
        a = self.act(self.layers[1](x))
        b = self.act(self.layers[2](a))
        x = a + b
        c = self.act(self.layers[3](x))
        d = self.act(self.layers[4](c))
        x = c + d
        e = self.act(self.layers[5](x))
        f = self.act(self.layers[6](e))
        x = e + f
        for l in self.layers[7:-1]:
            x = self.act(l(x))
        return F.softmax(self.layers[-1](x), dim=1)

# ============================================================
# EXPERIMENTS
# ============================================================
print("\nDataset difficulty check")
shallow = TwoLayerNet(torch.sigmoid)
train(shallow, 1)
print("Shallow accuracy:", evaluate(shallow))

print("\nBaseline deep model")
baseline = DeepNet(torch.sigmoid)
train(baseline, 1)
print("Baseline accuracy:", evaluate(baseline))

print("\nActivation experiments")
for name, act in [("Leaky ReLU", leaky_relu), ("Tanh", tanh)]:
    m = DeepNet(act)
    train(m, 1)
    print(name, "accuracy:", evaluate(m))

BEST_ACT = leaky_relu
BEST_BATCH = 32
BEST_MOM = 0.9

print("\nExtended model")
ext = ExtendedNet(BEST_ACT)
g_ext = train(ext, BEST_BATCH, BEST_MOM, track_grad=True)
print("Accuracy:", evaluate(ext), "Grad:", g_ext)

print("\nSkip model")
skip = SkipNetA(BEST_ACT)
g_skip = train(skip, BEST_BATCH, BEST_MOM, track_grad=True)
print("Accuracy:", evaluate(skip), "Grad:", g_skip)

print("\nExtra credit: weight decay")
for beta in WEIGHT_DECAY_VALUES:
    m = SkipNetA(BEST_ACT)
    train(m, BEST_BATCH, BEST_MOM, beta)
    print("Weight decay", beta, "accuracy:", evaluate(m))
