import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# python3 constructing_data_sets.py

# Linear Boundary
S = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]) # This is the S set we were given turned into a 1D tensor
x1, x2 = torch.meshgrid(S, S, indexing="ij") # This created a grid of all possible combinations of points in S
points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1) # This reshaped the grid into a list of 2D points
labels = (points[:, 1] > points[:, 0]).int().unsqueeze(1) # This created the labels based on the condition y > x

plt.figure(figsize=(6,6))
plt.scatter(points[:, 0], points[:, 1], c=labels.squeeze(), cmap='bwr', s=100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear Boundary Classification')
plt.axline((0, 0), slope=1, color='k', linestyle='--') # Line y = x
plt.legend(['Decision Boundary y=x','Class 0','Class 1'])
plt.grid()
plt.show()

# XOR Boundary
inputs = torch.tensor([[1.0,   1.0],
                       [1.0,  -1.0],
                       [-1.0, -1.0],
                       [-1.0,  1.0]]) # The input points for the XOR problem

labels = torch.tensor([0, 1, 0, 1]) # The corresponding labels for the XOR problem

plt.figure(figsize=(6,6))
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap='bwr', s=200)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Boundary Classification')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.legend(['x1=0','x2=0','Class 0','Class 1'])
plt.grid()
plt.show()