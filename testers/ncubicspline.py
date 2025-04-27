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

# Convert to PyTorch tensors
t_torch = torch.tensor(t, dtype=torch.float32)
c_torch = torch.tensor(c, dtype=torch.float32)
k_order = k

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

# Evaluate both implementations
x_scipy, y_scipy = splev(u_test, tck)
x_torch, y_torch = bspline_evaluate_2d(u_test_torch, t_torch, c_torch, k_order)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Compare curves
plt.subplot(2, 1, 1)
plt.plot(x_data, y_data, 'k.', label='Original Data', alpha=0.5)
plt.plot(x_scipy, y_scipy, 'b-', label='SciPy Spline', linewidth=2)
plt.plot(x_torch.numpy(), y_torch.numpy(), 'r--', label='PyTorch Spline', linewidth=2)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('2D B-Spline Interpolation Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# Plot 2: Differences
plt.subplot(2, 1, 2)
diff_x = x_torch.numpy() - x_scipy
diff_y = y_torch.numpy() - y_scipy
plt.plot(u_test, diff_x, 'r-', label='X difference', alpha=0.7)
plt.plot(u_test, diff_y, 'b-', label='Y difference', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Differences between PyTorch and SciPy Implementations')
plt.xlabel('Parameter u')
plt.ylabel('Difference')

# Add text with max differences
max_diff_x = np.max(np.abs(diff_x))
max_diff_y = np.max(np.abs(diff_y))
plt.text(0.02, 0.98, 
         f'Max abs difference: X: {max_diff_x:.2e}, Y: {max_diff_y:.2e}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()

# Save the plots
plt.savefig('/mnt/data/ashwin/neural_pmp/testers/testerplots/complex_2d_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot basis functions
plt.figure(figsize=(15, 5))
u_basis = torch.linspace(t[0], t[-1], 200)

# Plot several basis functions
colors = plt.cm.rainbow(np.linspace(0, 1, 8))
for i, color in zip(range(8), colors):
    basis = bspline_basis(u_basis, t_torch, i, k_order)
    plt.plot(u_basis.numpy(), basis.numpy(), '-', color=color, 
             label=f'Basis {i}', linewidth=2)

plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'2D B-Spline Basis Functions (Order {k_order})')
plt.xlabel('Parameter u')
plt.ylabel('Basis Value')

# Save the basis functions plot
plt.savefig('/mnt/data/ashwin/neural_pmp/testers/testerplots/complex_2d_basis_functions.png', dpi=300, bbox_inches='tight')
plt.close()