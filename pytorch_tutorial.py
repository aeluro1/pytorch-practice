import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    # z = x.to("cpu)")

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)


# x = torch.randn(3, requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# y = y.mean()
# y.backward()
# print(x.grad)