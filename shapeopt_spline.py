import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.interpolate import splprep, splev

#####################################
# B-spline Utility Functions
#####################################

def bspline_basis(x, t, i, k):
    """
    Compute the B-spline basis function B_{i,k}(x) given knot vector t.
    Using recursion with boundary checks.
    """
    if k == 0:
        # Order 0 basis functions are indicator functions for knot intervals.
        return ((x >= t[i]) & (x < t[i+1])).float()
    else:
        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]

        term1 = 0
        term2 = 0

        left_basis = bspline_basis(x, t, i, k-1)
        right_basis = bspline_basis(x, t, i+1, k-1)

        if denom1 != 0:
            term1 = (x - t[i]) / denom1 * left_basis
        if denom2 != 0:
            term2 = (t[i+k+1] - x) / denom2 * right_basis

        return term1 + term2

def bspline_evaluate_2d(u, t, c, k):
    """
    Evaluate the 2D B-spline at points u given knots t, coefficients c, and order k.
    c shape: (2, n_coeff)
    t shape: (len_knots,)
    """
    n_coeff = c.shape[1]
    result_x = torch.zeros_like(u)
    result_y = torch.zeros_like(u)
    
    for i in range(n_coeff):
        B_i = bspline_basis(u, t, i, k)
        result_x += c[0, i] * B_i
        result_y += c[1, i] * B_i
    
    return result_x, result_y

#####################################
# Neural Network Classes
#####################################

class Mlp(nn.Module):
    """Simple MLP network."""
    def __init__(self, input_dim, output_dim, layer_dims, activation='relu'):
        super(Mlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f'Activation {activation} not supported')
            
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in layer_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

class Encoder(nn.Module):
    """VAE-style encoder network."""
    def __init__(self, input_dim, output_dim, share_layer_dims, 
                 mean_layer_dims, logvar_layer_dims, activation='relu'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f'Activation {activation} not supported')
            
        # Shared layers
        self.share_layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in share_layer_dims:
            self.share_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            
        # Mean layers
        self.mean_layers = nn.ModuleList()
        mean_prev_dim = prev_dim
        for dim in mean_layer_dims:
            self.mean_layers.append(nn.Linear(mean_prev_dim, dim))
            mean_prev_dim = dim
        self.mean_layers.append(nn.Linear(mean_prev_dim, output_dim))
        
        # Logvar layers
        self.logvar_layers = nn.ModuleList()
        logvar_prev_dim = prev_dim
        for dim in logvar_layer_dims:
            self.logvar_layers.append(nn.Linear(logvar_prev_dim, dim))
            logvar_prev_dim = dim
        self.logvar_layers.append(nn.Linear(logvar_prev_dim, output_dim))
        
    def forward(self, x):
        # Shared layers
        for layer in self.share_layers:
            x = self.activation(layer(x))
            
        # Mean layers
        mean = x
        for layer in self.mean_layers[:-1]:
            mean = self.activation(layer(mean))
        mean = self.mean_layers[-1](mean)
        
        # Logvar layers
        logvar = x
        for layer in self.logvar_layers[:-1]:
            logvar = self.activation(layer(logvar))
        logvar = self.logvar_layers[-1](logvar)
        
        return mean, logvar

class HDNet(nn.Module):
    """(Forward) Hamiltonian dynamics network"""
    def __init__(self, hnet):
        super(HDNet, self).__init__()
        self.hnet = hnet
    
    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            q_p = torch.cat((q, p), dim=1)
            H = self.hnet(q_p)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dq, dp = torch.chunk(dH, 2, dim=1)
            return torch.cat((dp, -dq), dim=1)

class HDInverseNet(nn.Module):
    """Backward Hamiltonian dynamics network"""
    def __init__(self, hnet):
        super(HDInverseNet, self).__init__()
        self.hnet = hnet
    
    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            q_p = torch.cat((q, p), dim=1)
            H = self.hnet(q_p)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dq, dp = torch.chunk(dH, 2, dim=1)
            return torch.cat((-dp, dq), dim=1)

class HDVAE(nn.Module):
    """VAE-like forward-backward Hamiltonian dynamics architecture"""
    def __init__(self, adj_net, hnet, hnet_decoder, z_encoder, z_decoder, T):
        super(HDVAE, self).__init__()
        self.T = T
        self.adj_net = adj_net
        self.hd_net = HDNet(hnet)
        self.hd_inverse_net = HDInverseNet(hnet_decoder)
        self.z_encoder = z_encoder
        self.z_decoder = z_decoder
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, q):
        with torch.no_grad():
            times = [0, self.T]
            p = self.adj_net(q)
            qp = torch.cat((q, p), dim=1)
            qpt = odeint(self.hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
        
        mu, logvar = self.z_encoder(qpt)
        zhat = self.reparameterize(mu, logvar)
        qpt_hat = self.z_decoder(zhat)
        qp_hat = odeint(self.hd_inverse_net, qpt_hat, torch.tensor(times, requires_grad=True))[-1]
        
        return qp, qp_hat, qpt, qpt_hat, mu, logvar

#####################################
# Environment for B-spline Optimization
#####################################

class ContinuousEnv:
    """Base class for continuous control environments."""
    def __init__(self, q_dim=1, u_dim=1):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.eps = 1e-8
        self.id = np.eye(q_dim)
        
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))
    
    def f_u(self, q):
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))
    
    def L(self, q, u):
        return np.zeros(q.shape[0])
    
    def g(self, q):
        return np.zeros(q.shape[0])
    
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q+self.eps*self.id[i]) - self.g(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))
    
    def compute_area(self, q):
        raise NotImplementedError
    
    def compute_perimeter(self, q):
        raise NotImplementedError

class BSplineEnv(ContinuousEnv):
    """Environment where q represents B-spline coefficients for a 2D curve."""
    def __init__(self, n_coeff=10, k=3):
        # q_dim = 2*n_coeff (since we have x and y coefficients)
        q_dim = 2 * n_coeff
        super().__init__(q_dim, q_dim)
        self.n_coeff = n_coeff
        self.k = k
        # We create a knot vector t for the B-spline
        # A simple uniform knot vector that can handle n_coeff control points:
        # For a spline of order k with n_coeff points, we have n_coeff+k+1 knots
        self.t = np.linspace(0, 1, n_coeff - k + 1 + 2*k)  # Open uniform knots
        # The parameter space will be [t[0], t[-1]]
        self.torch_t = torch.tensor(self.t, dtype=torch.float32)
        
    def f(self, q, u):
        return u
    
    def f_u(self, q):
        N = q.shape[0]
        return np.repeat(np.eye(self.q_dim)[None, :, :], N, axis=0)
    
    def L(self, q, u):
        return np.sum(u**2, axis=1)
    
    def g(self, q):
        # Compute shape score = sqrt(area)/perimeter
        perimeter = self.compute_perimeter(q)
        area = self.compute_area(q)
        return np.sqrt(area)/(perimeter + self.eps)

    def _bspline_points(self, c_np, num_points=200):
        # c_np shape: (2*n_coeff,)
        c_np = c_np.reshape(2, self.n_coeff)
        u_eval = np.linspace(self.t[0], self.t[-1], num_points)
        u_torch = torch.tensor(u_eval, dtype=torch.float32)
        c_torch = torch.tensor(c_np, dtype=torch.float32)
        x_torch, y_torch = bspline_evaluate_2d(u_torch, self.torch_t, c_torch, self.k)
        x = x_torch.detach().numpy()
        y = y_torch.detach().numpy()
        # Close the curve by repeating the first point
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])
        return x, y

    def compute_area(self, q):
        # For each q, evaluate the B-spline and compute polygon area
        areas = []
        for i in range(q.shape[0]):
            x, y = self._bspline_points(q[i])
            # Shoelace formula
            area = 0.5 * np.abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
            areas.append(area)
        return np.array(areas)

    def compute_perimeter(self, q):
        perimeters = []
        for i in range(q.shape[0]):
            x, y = self._bspline_points(q[i])
            diffs = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
            perimeters.append(np.sum(diffs))
        return np.array(perimeters)
    
    def sample_q(self, num_examples, mode='train'):
        # Sample random control points around a baseline shape
        # For simplicity, let's start around a circle-like shape
        # and add noise. You can adjust this to sample different initial shapes.
        angles = np.linspace(0, 2*np.pi, self.n_coeff, endpoint=False)
        radius = 1.0
        base_x = radius * np.cos(angles)
        base_y = radius * np.sin(angles)
        
        # Add noise depending on mode
        if mode == 'train':
            noise_scale = 0.1
        else:
            noise_scale = 0.05
        
        qs = []
        for _ in range(num_examples):
            cx = base_x + noise_scale*np.random.randn(self.n_coeff)
            cy = base_y + noise_scale*np.random.randn(self.n_coeff)
            q = np.concatenate([cx, cy])
            qs.append(q)
        return np.stack(qs)

#####################################
# Utility Functions (Saving/Plotting)
#####################################

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_shape_plot(shapes, output_dir, env, shape_ids, stage="initial", title="Shape Plot"):
    """Save individual shape plots and a combined plot to specified directory."""
    ensure_directory(output_dir)
    stage_dir = os.path.join(output_dir, f"{stage}_shapes")
    ensure_directory(stage_dir)
    
    colors = ['b', 'r', 'g', 'c', 'm']
    
    # Save individual shapes
    for idx, q in enumerate(shapes):
        x, y = env._bspline_points(q)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.grid(True)
        
        color = colors[idx % len(colors)]
        ax.plot(x, y, '-', color=color, alpha=0.7)
        
        perimeter = env.compute_perimeter(q.reshape(1, -1))
        area = env.compute_area(q.reshape(1, -1))
        ax.set_title(f'Shape {shape_ids[idx]} ({stage})\nPerimeter: {perimeter[0]:.4f}, Area: {area[0]:.4f}')
        
        plt.savefig(os.path.join(stage_dir, f'shape_{shape_ids[idx]}.png'))
        plt.close(fig)
    
    # Save combined plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True)
    
    for idx, q in enumerate(shapes):
        x, y = env._bspline_points(q)
        color = colors[idx % len(colors)]
        ax.plot(x, y, '-', color=color, alpha=0.7, 
                label=f'Shape {shape_ids[idx]}')
    
    perimeter = env.compute_perimeter(shapes)
    area = env.compute_area(shapes)
    ax.set_title(f'All Shapes ({stage})\nMean Perimeter: {perimeter.mean():.4f}, Mean Area: {area.mean():.4f}')
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, f'{stage}_combined.png'))
    plt.close(fig)

def create_shape_animation(env, adj_net, hnet, initial_shape, output_path, num_frames=50):
    """Create an animation of shape optimization process for B-spline coefficients."""
    # Convert initial shape to tensor
    initial_shape_tensor = torch.tensor(initial_shape, dtype=torch.float32)
    
    # Get initial adjoint
    p = adj_net(initial_shape_tensor)
    qp = torch.cat((initial_shape_tensor, p), dim=1)
    
    # Create HDNet instance
    hd_net = HDNet(hnet=hnet)
    
    # Generate frames
    time_points = torch.linspace(0, 1.0, num_frames)
    
    with torch.no_grad():
        # Evolve the system and collect states
        qp_trajectory = odeint(hd_net, qp, time_points)
        
        # Extract shapes from q
        shapes = qp_trajectory[:, :, :env.q_dim].numpy()
        
        # Compute metrics for each frame
        perimeters = env.compute_perimeter(shapes.reshape(-1, env.q_dim)).reshape(num_frames, -1)
        areas = env.compute_area(shapes.reshape(-1, env.q_dim)).reshape(num_frames, -1)
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.grid(True)
        
        shape = shapes[frame]
        x, y = env._bspline_points(shape)
        line, = ax.plot(x, y, '-b', linewidth=2)
        
        metrics_text = f'Frame: {frame}/{num_frames-1}\n'
        metrics_text += f'Perimeter: {perimeters[frame,0]:.4f}\n'
        metrics_text += f'Area: {areas[frame,0]:.4f}\n'
        metrics_text += f'Shape Score: {np.sqrt(areas[frame,0])/perimeters[frame,0]:.4f}'
        
        ax.text(1.5, 1.5, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        return line,
    
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    
    anim.save(output_path, writer='pillow', fps=20)
    plt.close()

def animate_multiple_shapes(env, adj_net, hnet, num_shapes=5, output_dir="/tmp/animations"):
    """Create animations for multiple B-spline shapes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    initial_shapes = env.sample_q(num_shapes, mode='test')
    
    for i in range(num_shapes):
        output_path = os.path.join(output_dir, f'shape_{i+1}_optimization.gif')
        create_shape_animation(env, adj_net, hnet, 
                               initial_shapes[i:i+1], 
                               output_path)
        print(f"Created animation for shape {i+1}: {output_path}")

def visualize_training_results(env, adj_net, hnet, num_shapes=2):
    """Visualize evolution of multiple B-spline-based shapes."""
    output_dir = '/tmp/shape_outputs'
    ensure_directory(output_dir)
    
    shape_ids = list(range(1, num_shapes + 1))
    
    initial_shapes = torch.tensor(env.sample_q(num_shapes, mode='test'), dtype=torch.float)
    save_shape_plot(initial_shapes.numpy(), output_dir, env, shape_ids, "initial", "Initial Shapes")
    
    hd_net = HDNet(hnet=hnet)
    with torch.no_grad():
        p = adj_net(initial_shapes)
        qp = torch.cat((initial_shapes, p), dim=1)
        qp_final = odeint(hd_net, qp, torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True))[-1]
        final_shapes = qp_final[:, :env.q_dim].numpy()
    
    save_shape_plot(final_shapes, output_dir, env, shape_ids, "final", "Final Shapes")
    
    print(f"Visualization saved in {output_dir}")
    print(f"- Initial shapes saved in {output_dir}/initial_shapes/")
    print(f"- Final shapes saved in {output_dir}/final_shapes/")
    print(f"- Combined plots saved in {output_dir}/")

def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

#####################################
# Training Routines
#####################################

def train_phase_1(env, adj_net, hnet, qs, 
                  T1=1.0, control_coef=0.5, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, 
                  log_interval=50):
    hd_net = HDNet(hnet=hnet)
    optim = torch.optim.Adam(list(hnet.parameters()) + list(adj_net.parameters()), lr=lr)
    
    times = [0, T1]
    num_samples = qs.shape[0]
    
    print(f"Starting Phase 1 Training:")
    print(f"Number of samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    for i in range(num_epoch):
        epoch_adj_loss = 0
        epoch_ham_loss = 0
        epoch_total_loss = 0
        batches_in_epoch = 0
        
        print(f'\nEpoch {i+1}/{num_epoch}:')
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        
        for j in range(num_iter):
            indices = torch.randperm(num_samples)[:batch_size]
            q = q_dat[indices]
            
            # Get adjoint variables
            p = adj_net(q)
            qp = torch.cat((q, p), axis=1)

            # Evolve the system forward
            qp_t = odeint(hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
            qt, pt = torch.chunk(qp_t, 2, dim=1)

            # Convert to numpy for environment calculations
            q_np = q.detach().numpy()
            qt_np = qt.detach().numpy()
            p_np = p.detach().numpy()
            pt_np = pt.detach().numpy()

            # Calculate gradients using environment
            dg0 = torch.tensor(env.nabla_g(q_np), dtype=torch.float32)
            dg = torch.tensor(env.nabla_g(qt_np), dtype=torch.float32)

            # Compute adjoint loss
            adj_loss = alpha1 * F.smooth_l1_loss(p, dg0) + alpha2 * F.smooth_l1_loss(pt, -dg)

            # Compute Hamiltonian loss
            u = p_np / 2.0  # Control law
            h_pq_ref = np.einsum('ik,ik->i', p_np, u) - np.sum(u**2, axis=1)
            h_pq = hnet(qp)
            ham_loss = beta * F.smooth_l1_loss(h_pq, torch.tensor(h_pq_ref.reshape(-1, 1), dtype=torch.float32))
            
            # Total loss and backpropagation
            loss = adj_loss + ham_loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Accumulate metrics
            epoch_adj_loss += adj_loss.item()
            epoch_ham_loss += ham_loss.item()
            epoch_total_loss += loss.item()
            batches_in_epoch += 1

            # Logging
            if (j+1) % log_interval == 0:
                print(f'  Iteration {j+1}/{num_iter}:')
                print(f'    Adjoint Loss: {epoch_adj_loss/batches_in_epoch:.6f}')
                print(f'    Hamiltonian Loss: {epoch_ham_loss/batches_in_epoch:.6f}')
                print(f'    Total Loss: {epoch_total_loss/batches_in_epoch:.6f}')
                
                perimeter = env.compute_perimeter(qt_np)
                area = env.compute_area(qt_np)
                print(f'    Sample metrics - Perimeter: {perimeter.mean():.4f}, Area: {area.mean():.4f}')

                epoch_adj_loss = 0
                epoch_ham_loss = 0
                epoch_total_loss = 0
                batches_in_epoch = 0
def train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs, env, T2=1.0, beta=1.0, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, log_interval=50):
    hd_vae_net = HDVAE(adj_net, hnet, hnet_decoder, z_encoder, z_decoder, T2)
    optim = torch.optim.Adam(list(hnet_decoder.parameters()) + 
                             list(z_encoder.parameters()) +
                             list(z_decoder.parameters()), lr=lr)
    optim.zero_grad()
    
    num_samples = qs.shape[0]
    print(f"\nStarting Phase 2 Training:")
    print(f"Number of samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    for i in range(num_epoch):
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        batches_in_epoch = 0
        
        print(f'\nEpoch {i+1}/{num_epoch}:')
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        
        for j in range(num_iter):
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)]
            qp, qp_hat, qpt, qpt_hat, mu, logvar = hd_vae_net(q)
            
            recon_loss = F.smooth_l1_loss(qp, qp_hat) + F.smooth_l1_loss(qpt, qpt_hat)
            kl = beta * kl_loss(mu, logvar)
            loss = recon_loss + kl

            loss.backward()
            optim.step()
            optim.zero_grad()
            
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl.item()
            epoch_total_loss += loss.item()
            batches_in_epoch += 1

            if (j+1) % log_interval == 0:
                print(f'  Iteration {j+1}/{num_iter}:')
                print(f'    Reconstruction Loss: {epoch_recon_loss/batches_in_epoch:.6f}')
                print(f'    KL Loss: {epoch_kl_loss/batches_in_epoch:.6f}')
                print(f'    Total Loss: {epoch_total_loss/batches_in_epoch:.6f}')
                
                q_components = qp_hat[:, :qs.shape[1]].detach().numpy()
                recon_perimeter = env.compute_perimeter(q_components)
                recon_area = env.compute_area(q_components)
                print(f'    Sample metrics - Recon Perimeter: {recon_perimeter.mean():.4f}, Recon Area: {recon_area.mean():.4f}')
                
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                epoch_total_loss = 0
                batches_in_epoch = 0

def create_networks(q_dim, z_dim=8):
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, 
                  layer_dims=[64, 64], activation='tanh')
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, 
               layer_dims=[64, 64], activation='tanh')
    hnet_decoder = Mlp(input_dim=2*q_dim, output_dim=1, 
                       layer_dims=[64, 64], activation='tanh')
    z_encoder = Encoder(input_dim=2*q_dim, output_dim=z_dim,
                        share_layer_dims=[64, 32],
                        mean_layer_dims=[16],
                        logvar_layer_dims=[16],
                        activation='tanh')
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim,
                    layer_dims=[32, 64], activation='tanh')
    return adj_net, hnet, hnet_decoder, z_encoder, z_decoder

def main_training(env, env_name, qs, 
                 adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
                 T1=1, T2=1, control_coef=0.5, dynamic_hidden=False,
                 alpha1=1, alpha2=0.1, beta1=1, beta2=1,
                 num_epoch1=20, num_epoch2=20, num_iter1=20, num_iter2=20,
                 batch_size1=32, batch_size2=32, lr1=1e-3, lr2=1e-3,
                 log_interval1=50, log_interval2=50,
                 mode=0):
    # Train phase 1 
    print('\nTraining phase 1...')
    train_phase_1(env, adj_net, hnet, qs, 
                  T1, control_coef, dynamic_hidden, alpha1, alpha2, beta1,
                  num_epoch1, num_iter1, batch_size1, lr1, log_interval1)
    
    # Train phase 2 if mode >= 1
    if mode >= 1:
        print('\nTraining phase 2...')
        train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs, env,
                      T2, beta2, num_epoch2, num_iter2, batch_size2, lr2, log_interval2)

def main():
    n_coeff = 10
    k_order = 3
    env = BSplineEnv(n_coeff=n_coeff, k=k_order)
    
    q_dim = 2*n_coeff
    adj_net, hnet, hnet_decoder, z_encoder, z_decoder = create_networks(q_dim)
    
    num_examples = 200
    batch_size1 = 32
    batch_size2 = 32
    num_epoch1 = 30
    num_epoch2 = 20
    num_iter1 = 50
    num_iter2 = 50
    lr1 = 1e-3
    lr2 = 1e-3
    log_interval1 = 10
    log_interval2 = 10
    
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    main_training(env, "bspline_opt", q_samples,
                  adj_net, hnet, hnet_decoder, z_decoder, z_encoder,
                  T1=1.0, T2=1.0, control_coef=0.5, dynamic_hidden=False,
                  alpha1=1.0, alpha2=0.1, beta1=1.0, beta2=1.0,
                  num_epoch1=num_epoch1, num_epoch2=num_epoch2,
                  num_iter1=num_iter1, num_iter2=num_iter2,
                  batch_size1=batch_size1, batch_size2=batch_size2,
                  lr1=lr1, lr2=lr2,
                  log_interval1=log_interval1, log_interval2=log_interval2,
                  mode=1)

    print("\nCreating visualizations...")
    visualize_training_results(env, adj_net, hnet, num_shapes=5)
    
    print("\nGenerating shape optimization animations...")
    animate_multiple_shapes(env, adj_net, hnet, num_shapes=2)

if __name__ == "__main__":
    main()
