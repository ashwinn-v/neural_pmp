import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

###############################################################################
# Neural Network Components
# These networks are used to approximate various functions from the PMP-based
# optimal control formulation. For example:
# - adj_net approximates the adjoint variables p(q).
# - hnet and hnet_decoder approximate the Hamiltonian function H(q, p).
# - z_encoder and z_decoder approximate reduced-dimensional representations of
#   state variables, potentially useful for more complex shape dynamics.
###############################################################################

class Mlp(nn.Module):
    """Simple MLP network.
    
    This is a generic MLP used to represent functions such as:
    - The adjoint mapping q -> p (through adj_net)
    - The Hamiltonian H(q,p) (through hnet)
    
    The activation and layer structure are configurable.
    """
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
    """VAE-style encoder network.
    
    Used to produce a latent representation (mean and log-variance)
    of the final shape (q,p). This can be useful for regularization
    or learning a reduced representation of shape states.
    """
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

###############################################################################
# Hamiltonian Dynamics Networks
#
# The PMP states the time evolution of the state q(t) and adjoint p(t) via
# Hamiltonian equations. If we consider H(q,p), then we have:
#   dq/dt = ∂H/∂p
#   dp/dt = -∂H/∂q
#
# Below, we define networks (HDNet and HDInverseNet) that simulate forward and
# backward Hamiltonian flows given a learned Hamiltonian H.
###############################################################################

class HDNet(nn.Module):
    """(Forward) Hamiltonian dynamics network.
    
    Given H(q,p), we compute:
        dq/dt = ∂H/∂p,
        dp/dt = -∂H/∂q.
    """
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
    """Backward Hamiltonian dynamics network.
    
    This is useful for mapping terminal conditions backward in time.
    """
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
            # Reversed dynamics (for inverse flow)
            return torch.cat((-dp, dq), dim=1)

###############################################################################
# HDVAE combines forward and backward Hamiltonian flows with VAE encoders/decoders.
# This is a more complex architecture that attempts to regularize and learn
# a latent representation of shape trajectories. It aims to ensure the terminal
# conditions are matched and that the solution fits into a low-dimensional manifold.
###############################################################################

class HDVAE(nn.Module):
    """VAE-like forward-backward Hamiltonian dynamics architecture.
    
    Integrates the PMP framework with latent variable models:
    - Forward pass: q -> p -> (q,p) evolves via HDNet
    - Encode final state into z
    - Decode back into shape and run backward via HDInverseNet
    """
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
        # Forward flow from q to q(T)
        with torch.no_grad():
            times = [0, self.T]
            p = self.adj_net(q)
            qp = torch.cat((q, p), dim=1)
            qpt = odeint(self.hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
        
        # Encode final state (q(1), p(1)) into latent z
        mu, logvar = self.z_encoder(qpt)
        zhat = self.reparameterize(mu, logvar)
        
        # Decode z back to (q(1), p(1)) and run inverse dynamics
        qpt_hat = self.z_decoder(zhat)
        qp_hat = odeint(self.hd_inverse_net, qpt_hat, torch.tensor(times, requires_grad=True))[-1]
        
        return qp, qp_hat, qpt, qpt_hat, mu, logvar

###############################################################################
# Continuous Environment for Shape Optimization
#
# The shape space is represented as q ∈ M, where q is a discretized boundary
# (e.g., polygon vertices) of a shape. Controls u = v(t) deform the shape over time.
#
# The cost function includes:
# - Running cost: L(q(t), v(t)) ~ ||v(t)||^2
# - Terminal cost: g(q(1)) = sqrt(area(q(1)))/perimeter(q(1))
#
# The environment provides:
# - State f(q,u) = u (the shape evolves by the velocity field u)
# - Running cost L(q,u)
# - Terminal cost g(q)
#
# This setup aligns with the PMP framework:
#   dq/dt = u
#   Minimizing ∫ L(q,u) dt + g(q(1))
###############################################################################

class ContinuousEnv:
    """Base class for continuous control environments.
    
    This abstract environment defines:
    - f(q, u): state dynamics
    - f_u(q): partial derivatives wrt u if needed
    - L(q, u): running cost
    - g(q): terminal cost
    - nabla_g(q): gradient of terminal cost
    
    Here, q_dim and u_dim represent dimensions of state and control.
    """
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
        # Numerically approximate gradient of g at q
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q+self.eps*self.id[i]) - self.g(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))

class ShapeOptEnv(ContinuousEnv):
    """2D shape optimization environment.
    
    Here q represents the coordinates of vertices of a polygonal shape.
    The terminal cost g(q) = sqrt(area(q)) / perimeter(q) encourages certain
    shape properties (like area-to-perimeter ratio).
    The running cost L(q,u) = ||u||^2 encourages minimal deformation effort.
    
    This maps directly to our shape optimization problem discussed before:
    - q(t) = shape at time t
    - u(t) = velocity field controlling shape deformation
    - Minimize integral of L(q,u) dt + g(q(1))
    """
    def __init__(self, q_dim=10, u_dim=10):
        super().__init__(q_dim, u_dim)
        
    def compute_area(self, q):
        points = q.reshape(-1, self.q_dim//2, 2)
        closed_points = np.concatenate([points, points[:, :1, :]], axis=1)
        x = closed_points[:, :, 0]
        y = closed_points[:, :, 1]
        area = 0.5 * np.abs(np.sum(x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1], axis=1))
        return area
        
    def compute_perimeter(self, q):
        points = q.reshape(-1, self.q_dim//2, 2)
        closed_points = np.concatenate([points, points[:, :1, :]], axis=1)
        diffs = closed_points[:, 1:, :] - closed_points[:, :-1, :]
        distances = np.sqrt(np.sum(diffs**2, axis=2))
        return np.sum(distances, axis=1)
    
    def f(self, q, u):
        # dq/dt = u
        return u
    
    def f_u(self, q):
        N = q.shape[0]
        return np.repeat(np.eye(self.q_dim)[None, :, :], N, axis=0)
    
    def L(self, q, u):
        # Running cost: norm(u)^2
        return np.sum(u**2, axis=1)
    
    def g(self, q):
        # Terminal cost: sqrt(area(q))/perimeter(q)
        perimeter = self.compute_perimeter(q)
        area = self.compute_area(q)
        return np.sqrt(area) / (perimeter + self.eps)
    
    # Methods to sample shapes and interpolate vertices
    def _create_square(self, size=1.0, noise_scale=0.0):
        vertices = np.array([
            [-size, -size], [size, -size], 
            [size, size], [-size, size]
        ])
        if noise_scale > 0:
            vertices += noise_scale * np.random.randn(*vertices.shape)
        return vertices
    
    def _create_rectangle(self, width=1.0, height=0.5, noise_scale=0.0):
        vertices = np.array([
            [-width, -height], [width, -height],
            [width, height], [-width, height]
        ])
        if noise_scale > 0:
            vertices += noise_scale * np.random.randn(*vertices.shape)
        return vertices
    
    def _create_triangle(self, size=1.0, noise_scale=0.0):
        vertices = np.array([
            [0, size],
            [size * np.cos(7*np.pi/6), size * np.sin(7*np.pi/6)],
            [size * np.cos(11*np.pi/6), size * np.sin(11*np.pi/6)]
        ])
        if noise_scale > 0:
            vertices += noise_scale * np.random.randn(*vertices.shape)
        return vertices
    
    def _create_pentagon(self, size=1.0, noise_scale=0.0):
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        vertices = size * np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        if noise_scale > 0:
            vertices += noise_scale * np.random.randn(*vertices.shape)
        return vertices
    
    def _interpolate_vertices(self, vertices, n_vertices):
        if len(vertices) == n_vertices:
            return vertices
            
        closed_vertices = np.vstack([vertices, vertices[0]])
        diffs = np.diff(closed_vertices, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cum_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cum_lengths[-1]
        
        desired_points = np.linspace(0, total_length, n_vertices, endpoint=False)
        
        new_vertices = []
        for t in desired_points:
            idx = np.searchsorted(cum_lengths, t) - 1
            idx = max(0, min(idx, len(vertices)-1))
            alpha = (t - cum_lengths[idx]) / segment_lengths[idx]
            alpha = max(0, min(alpha, 1))
            point = vertices[idx] + alpha * (vertices[(idx+1)%len(vertices)] - vertices[idx])
            new_vertices.append(point)
            
        return np.array(new_vertices)
    
    def sample_q(self, num_examples, mode='train'):
        n_vertices = self.q_dim // 2
        shapes = []
        
        if mode == 'train':
            noise_scale = 0.05
        else:
            noise_scale = 0.02
            
        for _ in range(num_examples):
            shape_type = np.random.choice(['square', 'rectangle', 'triangle', 'pentagon'])
            
            if shape_type == 'square':
                size = np.random.uniform(0.8, 1.2)
                vertices = self._create_square(size, noise_scale)
            elif shape_type == 'rectangle':
                width = np.random.uniform(0.8, 1.2)
                height = np.random.uniform(0.4, 0.8)
                vertices = self._create_rectangle(width, height, noise_scale)
            elif shape_type == 'triangle':
                size = np.random.uniform(0.8, 1.2)
                vertices = self._create_triangle(size, noise_scale)
            else:  # pentagon
                size = np.random.uniform(0.8, 1.2)
                vertices = self._create_pentagon(size, noise_scale)
                
            vertices = self._interpolate_vertices(vertices, n_vertices)
            max_radius = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
            vertices = vertices / max_radius
            shapes.append(vertices)
        
        q = np.stack(shapes).reshape(num_examples, self.q_dim)
        return q

###############################################################################
# Utility functions for plotting and animation
###############################################################################

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_shape_plot(shapes, output_dir, env, shape_ids, stage="initial", title="Shape Plot"):
    """Save individual shape plots and a combined plot to specified directory."""
    ensure_directory(output_dir)
    stage_dir = os.path.join(output_dir, f"{stage}_shapes")
    ensure_directory(stage_dir)
    
    vertices = shapes.reshape(-1, env.q_dim//2, 2)
    colors = ['b', 'r', 'g', 'c', 'm']
    
    # Save individual shapes
    for idx, shape in enumerate(vertices):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.grid(True)
        
        closed_shape = np.vstack([shape, shape[0]])
        color = colors[idx % len(colors)]
        ax.plot(closed_shape[:, 0], closed_shape[:, 1], '-', color=color, alpha=0.7)
        
        perimeter = env.compute_perimeter(shape.reshape(1, -1))
        area = env.compute_area(shape.reshape(1, -1))
        ax.set_title(f'Shape {shape_ids[idx]} ({stage})\nPerimeter: {perimeter[0]:.4f}, Area: {area[0]:.4f}')
        
        plt.savefig(os.path.join(stage_dir, f'shape_{shape_ids[idx]}.png'))
        plt.close(fig)
    
    # Save combined plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True)
    
    for idx, shape in enumerate(vertices):
        closed_shape = np.vstack([shape, shape[0]])
        color = colors[idx % len(colors)]
        ax.plot(closed_shape[:, 0], closed_shape[:, 1], '-', color=color, alpha=0.7, 
                label=f'Shape {shape_ids[idx]}')
    
    perimeter = env.compute_perimeter(shapes)
    area = env.compute_area(shapes)
    ax.set_title(f'All Shapes ({stage})\nMean Perimeter: {perimeter.mean():.4f}, Mean Area: {area.mean():.4f}')
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, f'{stage}_combined.png'))
    plt.close(fig)

def create_shape_animation(env, adj_net, hnet, initial_shape, output_path, num_frames=50):
    """Create an animation of shape optimization process.
    
    Shows how the shape evolves over time under the learned control (via HDNet),
    illustrating the PMP-based solution where q(t) evolves to minimize cost.
    """
    initial_shape_tensor = torch.tensor(initial_shape, dtype=torch.float32)
    
    # Get initial adjoint p(q)
    p = adj_net(initial_shape_tensor)
    qp = torch.cat((initial_shape_tensor, p), dim=1)
    
    # Create HDNet instance for forward simulation
    hd_net = HDNet(hnet=hnet)
    
    # Evolve the system over time
    time_points = torch.linspace(0, 1.0, num_frames)
    with torch.no_grad():
        qp_trajectory = odeint(hd_net, qp, time_points)
        shapes = qp_trajectory[:, :, :env.q_dim].numpy()
        
        # Compute metrics for each frame
        perimeters = np.array([env.compute_perimeter(shape) for shape in shapes])
        areas = np.array([env.compute_area(shape) for shape in shapes])
        
    # Create animation plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Get current shape
        shape = shapes[frame].reshape(-1, env.q_dim//2, 2)[0]
        closed_shape = np.vstack([shape, shape[0]])
        
        # Plot shape
        line, = ax.plot(closed_shape[:, 0], closed_shape[:, 1], '-b', linewidth=2)
        
        # Add metrics text
        metrics_text = f'Frame: {frame}/{num_frames-1}\n'
        metrics_text += f'Perimeter: {perimeters[frame][0]:.4f}\n'
        metrics_text += f'Area: {areas[frame][0]:.4f}\n'
        metrics_text += f'Shape Score: {np.sqrt(areas[frame][0])/perimeters[frame][0]:.4f}'
        
        ax.text(1.5, 1.5, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        return line,
    
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close()

def animate_multiple_shapes(env, adj_net, hnet, num_shapes=5, output_dir="/mnt/data/ashwin/neural_pmp/animations"):
    """Create animations for multiple shapes."""
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
    """Visualize evolution of sample shapes. 
       
    Shows initial and final shapes after running the learned PDE-based
    PMP-driven deformation. This helps confirm that shapes evolve towards
    optimal configurations.
    """
    output_dir = '/mnt/data/ashwin/neural_pmp/shape_outputs1'
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

###############################################################################
# Training routines:
# Two-phase training:
#
# Phase 1: Train adj_net and hnet to approximate PMP conditions:
#   - Enforce conditions on adjoint variables p and the Hamiltonian.
#
# Phase 2: Train hnet_decoder, z_encoder, z_decoder to refine solutions with 
#   latent representations and backward-forward consistency.
###############################################################################

def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

def train_phase_1(env, adj_net, hnet, qs, 
                  T1=1.0, control_coef=0.5, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, 
                  log_interval=50):
    """Phase 1 Training:
    
    - Trains adj_net and hnet to satisfy PMP conditions.
    - adj_net learns p = p(q), approximating the adjoint solution.
    - hnet learns H(q,p) to ensure correct Hamiltonian structure and 
      boundary conditions (p(1) = -dg/dq).
    
    We use the shape environment to sample q, compute target gradients of g,
    and encourage consistency with PMP conditions.
    """
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
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)]
            q_np = q.detach().numpy()
            
            p = adj_net(q)
            p_np = p.detach().numpy()
            qp = torch.cat((q, p), axis=1)

            # Forward integrate q,p via Hamiltonian dynamics
            qp_t = odeint(hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
            qt, pt = torch.chunk(qp_t, 2, dim=1)
            qt_np = qt.detach().numpy()
            pt_np = pt.detach().numpy()

            # Compute gradients of terminal cost
            dg0 = torch.tensor(env.nabla_g(q_np))
            dg = torch.tensor(env.nabla_g(qt_np))

            # Adjoint loss encourages correct boundary conditions:
            # p(q0) ~ dg(q0), pt(q1) ~ -dg(q1)
            adj_loss = alpha1 * F.smooth_l1_loss(p, dg0) + alpha2 * F.smooth_l1_loss(pt, -dg)

            # Control guess and Hamiltonian references
            u = p_np / 2.0
            h_pq_ref = np.einsum('ik,ik->i', p_np, u) - np.sum(u**2, axis=1)

            h_pq = hnet(qp)
            ham_loss = beta * F.smooth_l1_loss(h_pq, torch.tensor(h_pq_ref.reshape(-1, 1), dtype=torch.float))
            
            loss = adj_loss + ham_loss

            loss.backward()
            optim.step()
            optim.zero_grad()
            
            epoch_adj_loss += adj_loss.item()
            epoch_ham_loss += ham_loss.item()
            epoch_total_loss += loss.item()
            batches_in_epoch += 1

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
    """Phase 2 Training:
    
    Refining solution using VAE-like approach:
    - Use HDVAE to ensure forward-backward consistency in (q,p)
    - Encourage latent codes to represent trajectories well
    - Introduce KL loss for stable latent space
    
    This phase fine-tunes the learned dynamics to be consistent forward and backward.
    """
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
    """Create the necessary neural networks for the PMP-based shape optimization.
    
    - adj_net: approximates adjoint p(q)
    - hnet: Hamiltonian H(q,p)
    - hnet_decoder: for backward/decoder Hamiltonian 
    - z_encoder, z_decoder: latent VAE components
    
    The chosen architectures (layer_dims, activation) are empirical.
    """
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
    # Phase 1: Train adj_net and hnet
    print('\nTraining phase 1 (PMP conditions enforcement)...')
    train_phase_1(env, adj_net, hnet, qs, 
                  T1, control_coef, dynamic_hidden, alpha1, alpha2, beta1,
                  num_epoch1, num_iter1, batch_size1, lr1, log_interval1)
    
    # Phase 2: If mode >= 1, train decoder and latent nets
    if mode >= 1:
        print('\nTraining phase 2 (Forward-backward consistency with VAE)...')
        train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs, env,
                      T2, beta2, num_epoch2, num_iter2, batch_size2, lr2, log_interval2)

def main():
    q_dim = 20  # For example, 10 vertices with x,y coordinates
    env = ShapeOptEnv(q_dim=q_dim, u_dim=q_dim)
    
    adj_net, hnet, hnet_decoder, z_encoder, z_decoder = create_networks(q_dim)
    
    # Number of training examples
    num_examples = 1000
    # Training parameters
    batch_size1 = 32
    batch_size2 = 32
    num_epoch1 = 30
    num_epoch2 = 20
    num_iter1 = 100
    num_iter2 = 100
    lr1 = 1e-3
    lr2 = 1e-3
    log_interval1 = 10
    log_interval2 = 10
    
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Run main training with mode=1 to run both phases
    main_training(env, "shape_opt", q_samples,
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
    # Visualizations of initial and final shapes
    visualize_training_results(env, adj_net, hnet, num_shapes=20)
    
    # Animations of shape optimization
    print("\nGenerating shape optimization animations...")
    animate_multiple_shapes(env, adj_net, hnet, num_shapes=10)

if __name__ == "__main__":
    main()
