import torch
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# Create a complex 2D shape (figure-8 with perturbations)
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 100)
x_data = np.sin(2*t) + 0.2*np.sin(5*t) + 0.1*np.random.randn(100)
y_data = np.sin(t) + 0.2*np.cos(3*t) + 0.1*np.random.randn(100)

# Fit a 2D B-spline using scipy
tck, u = splprep([x_data, y_data], k=3, s=0.1)  # s is smoothing factor
t, c, k = tck

t_torch = torch.tensor(t, dtype=torch.float32)
c_torch = torch.tensor(c, dtype=torch.float32)
k_order = k

t_torch.requires_grad = False
c_torch.requires_grad = True

def bspline_basis(x, t, i, k):
    """
    Compute the B-spline basis function B_{i,k}(x) given knot vector t.
    Using recursion with boundary checks.
    """
    if i < 0 or i >= len(t)-1:
        return torch.zeros_like(x)

    if k == 0:
        if i+1 >= len(t):
            return torch.zeros_like(x)
        return ((x >= t[i]) & (x < t[i+1])).float()
    else:
        if i+k >= len(t) or i+k+1 >= len(t) or i+1 >= len(t):
            return torch.zeros_like(x)

        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]

        term1 = (x - t[i]) / denom1 if denom1 != 0 else torch.zeros_like(x)
        term2 = (t[i+k+1] - x) / denom2 if denom2 != 0 else torch.zeros_like(x)

        left_basis = bspline_basis(x, t, i, k-1)
        right_basis = bspline_basis(x, t, i+1, k-1)

        return term1 * left_basis + term2 * right_basis

def bspline_evaluate_2d(u, t, c, k):
    """
    Evaluate the 2D B-spline at points u given knots t, coefficients c, and order k.
    """
    n_coeff = c.shape[1]
    result_x = torch.zeros_like(u)
    result_y = torch.zeros_like(u)
    
    for i in range(n_coeff):
        B_i = bspline_basis(u, t, i, k)
        result_x += c[0, i] * B_i
        result_y += c[1, i] * B_i
    
    return result_x, result_y

# Test points for evaluation
u_test = np.linspace(t[0], t[-1], 200)
u_test_torch = torch.tensor(u_test, dtype=torch.float32)


x_torch, y_torch = bspline_evaluate_2d(u_test_torch, t_torch, c_torch, k_order)

print(c_torch)