# Modified from: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#

import torch
import math

manual = False
auto = True


torch.manual_seed(42)

# Manual implementation of a NN to approximate a sine wave
if manual:
    dtype = torch.float
    device = torch.device("cpu")

    # Create random input and output data
    x = torch.linspace(-2 * math.pi, 2 * math.pi, 4000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)
    e = torch.randn((), device=device, dtype=dtype)
    f = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-8

    for t in range(int(1e5)):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 1000 == 99:
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()
        grad_e = (grad_y_pred * x ** 4).sum()
        grad_f = (grad_y_pred * x ** 5).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate / 1 * grad_b
        c -= learning_rate / 1 * grad_c
        d -= learning_rate / 1 * grad_d
        e -= learning_rate / 1e2 * grad_e
        f -= learning_rate / 1e3 * grad_f


    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3 + {e.item()} x^4 + {f.item()} x^5')

# Automatic implementation from torch
if auto:
    dtype = torch.float
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
    y = torch.sin(x)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.randn((), dtype=dtype, requires_grad=True)
    b = torch.randn((), dtype=dtype, requires_grad=True)
    c = torch.randn((), dtype=dtype, requires_grad=True)
    d = torch.randn((), dtype=dtype, requires_grad=True)
    e = torch.randn((), dtype=dtype, requires_grad=True)
    f = torch.randn((), dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(15000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad
            e -= learning_rate / 100 * e.grad
            f -= learning_rate / 1000 * f.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None
            e.grad = None
            f.grad = None

    print(f'Result: y = {a.item()} + {b.item()} * x + {c.item()} * x^2 + {d.item()} * x^3 + {e.item()} * x^4 + {f.item()} * x^5')