import numpy as np
import torch
from torch import nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)

# Model prediction
def forward(x):
    return w * x

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate) # Stochastic gradient descent

for epoch in range(n_iters):

    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass, dl/dw
    l.backward()

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

    print(f"Prediction before training: f(5) = {forward(5):.3f}")
