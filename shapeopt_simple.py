import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchdiffeq import odeint_adjoint as odeint
from common_nets import Mlp, Encoder

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_shape_evolution(env, adj_net, hnet, initial_shape, T=1.0, num_frames=50):
    """
    Create an animation of shape evolution over time.
    """
    hd_net = HDNet(hnet=hnet)
    times = np.linspace(0, T, num_frames)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Initialize shape
    q = initial_shape
    p = adj_net(q)
    qp = torch.cat((q, p), dim=1)
    
    # Compute evolution
    evolved_shapes = []
    with torch.no_grad():
        for t in times:
            # Create time tensor with proper ordering
            t_tensor = torch.tensor([0.0, float(t)], dtype=torch.float32, requires_grad=True)
            qp_t = odeint(hd_net, qp, t_tensor)[-1]
            qt, _ = torch.chunk(qp_t, 2, dim=1)
            evolved_shapes.append(qt.detach().numpy())
    
    evolved_shapes = np.array(evolved_shapes)
    
    # Animation function
    def animate(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Get current shape
        current_shape = evolved_shapes[frame]
        
        # Reshape to get vertices
        vertices = current_shape.reshape(-1, env.q_dim//2, 2)
        
        # Plot each shape in the batch with different colors
        colors = ['b', 'r', 'g', 'c', 'm']  # Add more colors if needed
        for idx, shape in enumerate(vertices):
            # Close the polygon by adding first vertex at the end
            closed_shape = np.vstack([shape, shape[0]])
            color = colors[idx % len(colors)]
            ax.plot(closed_shape[:, 0], closed_shape[:, 1], '-', color=color, alpha=0.7)
            
        # Add metrics
        perimeter = env.compute_perimeter(current_shape)
        area = env.compute_area(current_shape)
        ax.set_title(f'Time: {times[frame]:.2f}\nPerimeter: {perimeter.mean():.4f}, Area: {area.mean():.4f}')
        
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=50, blit=True)
    
    return anim

def save_evolution_video(anim, filename='shape_evolution.mp4'):
    """Save the animation to a video file."""
    anim.save(filename, writer='ffmpeg', fps=30)

def visualize_training_results(env, adj_net, hnet, num_shapes=5):
    """Visualize the evolution of multiple sample shapes."""
    # Generate some sample initial shapes
    initial_shapes = torch.tensor(env.sample_q(num_shapes, mode='test'), 
                                dtype=torch.float)
    
    # Create and save animation
    print("Generating animation...")
    anim = visualize_shape_evolution(env, adj_net, hnet, initial_shapes)
    print("Saving animation to file...")
    save_evolution_video(anim)
    print("Animation saved successfully!")
    
    return anim
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
            # Use backward dynamics: f = (-h_p, h_q)
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
            # Use backward dynamics: f = (-h_p, h_q)
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

def kl_loss(mu, logvar):
    """KL divergence loss"""
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

def train_phase_1(env, adj_net, hnet, qs, 
                  T1=1.0, control_coef=0.5, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, 
                  log_interval=50):
    """Train phase 1 of Neural PMP with detailed progress monitoring"""
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
            # Get batch
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)]
            q_np = q.detach().numpy()
            
            # Forward pass
            p = adj_net(q)
            p_np = p.detach().numpy()
            qp = torch.cat((q, p), axis=1)

            # Evolve system
            qp_t = odeint(hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
            qt, pt = torch.chunk(qp_t, 2, dim=1)
            qt_np = qt.detach().numpy()

            # Compute adjoint losses
            dg0 = torch.tensor(env.nabla_g(q_np))
            dg = torch.tensor(env.nabla_g(qt_np))
            adj_loss = alpha1 * F.smooth_l1_loss(p, dg0) + alpha2 * F.smooth_l1_loss(pt, dg)

            # Compute Hamiltonian losses
            u = (1.0/control_coef)*np.einsum('ijk,ij->ik', env.f_u(q_np), -p_np)
            
            if dynamic_hidden:
                qp_dot = hd_net(0, qp)
                qdot, _ = torch.chunk(qp_dot, 2, dim=1)
                qdot_np = qdot.detach().numpy()
                h_pq_ref = np.einsum('ik,ik->i', p_np, qdot_np) + env.L(q_np, u)
            else:
                h_pq_ref = np.einsum('ik,ik->i', p_np, env.f(q_np, u)) + env.L(q_np, u)
                
            h_pq = hnet(qp)
            ham_loss = beta * F.smooth_l1_loss(h_pq, torch.tensor(h_pq_ref.reshape(-1, 1)))

            # Total loss
            loss = adj_loss + ham_loss

            # Backward pass
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Accumulate metrics
            epoch_adj_loss += adj_loss.item()
            epoch_ham_loss += ham_loss.item()
            epoch_total_loss += loss.item()
            batches_in_epoch += 1

            if (j+1) % log_interval == 0:
                print(f'  Iteration {j+1}/{num_iter}:')
                print(f'    Adjoint Loss: {epoch_adj_loss/batches_in_epoch:.6f}')
                print(f'    Hamiltonian Loss: {epoch_ham_loss/batches_in_epoch:.6f}')
                print(f'    Total Loss: {epoch_total_loss/batches_in_epoch:.6f}')
                
                # Compute some shape metrics for a sample
                perimeter = env.compute_perimeter(qt_np)
                area = env.compute_area(qt_np)
                print(f'    Sample metrics - Perimeter: {perimeter.mean():.4f}, Area: {area.mean():.4f}')
                
                epoch_adj_loss = 0
                epoch_ham_loss = 0
                epoch_total_loss = 0
                batches_in_epoch = 0

def train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs, env, T2=1.0, beta=1.0, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, log_interval=50):
    """Train phase 2 of Neural PMP with detailed progress monitoring"""
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
            # Get batch and forward pass
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)]
            qp, qp_hat, qpt, qpt_hat, mu, logvar = hd_vae_net(q)
            
            # Compute losses
            recon_loss = F.smooth_l1_loss(qp, qp_hat) + F.smooth_l1_loss(qpt, qpt_hat)
            kl = beta * kl_loss(mu, logvar)
            loss = recon_loss + kl

            # Backward pass
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Accumulate metrics
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl.item()
            epoch_total_loss += loss.item()
            batches_in_epoch += 1

            if (j+1) % log_interval == 0:
                print(f'  Iteration {j+1}/{num_iter}:')
                print(f'    Reconstruction Loss: {epoch_recon_loss/batches_in_epoch:.6f}')
                print(f'    KL Loss: {epoch_kl_loss/batches_in_epoch:.6f}')
                print(f'    Total Loss: {epoch_total_loss/batches_in_epoch:.6f}')
                
                # Extract q components for metrics
                q_components = qp_hat[:, :qs.shape[1]].detach().numpy()
                recon_perimeter = env.compute_perimeter(q_components)
                recon_area = env.compute_area(q_components)
                print(f'    Sample metrics - Recon Perimeter: {recon_perimeter.mean():.4f}, Recon Area: {recon_area.mean():.4f}')
                
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                epoch_total_loss = 0
                batches_in_epoch = 0

def main_training(env, env_name, qs, 
                 adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
                 T1=1, T2=1, control_coef=0.5, dynamic_hidden=False,
                 alpha1=1, alpha2=0.1, beta1=1, beta2=1,
                 num_epoch1=20, num_epoch2=20, num_iter1=20, num_iter2=20,
                 batch_size1=32, batch_size2=32, lr1=1e-3, lr2=1e-3,
                 log_interval1=50, log_interval2=50,
                 mode=0):
    """Main training function including both phases"""
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
class ContinuousEnv:
    """Base class for continuous control environments used in PMP."""
    def __init__(self, q_dim=1, u_dim=1):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.eps = 1e-8
        self.id = np.eye(q_dim)
        
    def f(self, q, u):
        """System dynamics."""
        return np.zeros((q.shape[0], self.q_dim))
    
    def f_u(self, q):
        """Partial derivative of dynamics wrt control."""
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))
    
    def L(self, q, u):
        """Running cost."""
        return np.zeros(q.shape[0])
    
    def g(self, q):
        """Terminal cost."""
        return np.zeros(q.shape[0])
    
    def nabla_g(self, q):
        """Gradient of terminal cost."""
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q+self.eps*self.id[i])-self.g(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    def sample_q(self, num_examples, mode='train'):
        """Sample initial states."""
        return np.zeros((num_examples, self.q_dim))

class ShapeOptEnv(ContinuousEnv):
    """2D shape optimization environment."""
    def __init__(self, q_dim=10, u_dim=10, target_area=1.0):
        super().__init__(q_dim, u_dim)
        self.target_area = target_area
        self.area_weight = 100.0  # Weight for area constraint
        
    def compute_area(self, q):
        """Compute area of polygon defined by points in q."""
        # Reshape q into x,y coordinates
        points = q.reshape(-1, self.q_dim//2, 2)
        # Add first point to end to close the polygon
        closed_points = np.concatenate([points, points[:,:1,:]], axis=1)
        # Compute area using shoelace formula
        x = closed_points[:,:,0]
        y = closed_points[:,:,1]
        area = 0.5 * np.abs(np.sum(x[:,:-1] * y[:,1:] - x[:,1:] * y[:,:-1], axis=1))
        return area
        
    def compute_perimeter(self, q):
        """Compute perimeter of polygon defined by points in q."""
        points = q.reshape(-1, self.q_dim//2, 2)
        # Add first point to end to close the polygon
        closed_points = np.concatenate([points, points[:,:1,:]], axis=1)
        # Compute distances between consecutive points
        diffs = closed_points[:,1:,:] - closed_points[:,:-1,:]
        distances = np.sqrt(np.sum(diffs**2, axis=2))
        return np.sum(distances, axis=1)
    
    def f(self, q, u):
        """Shape evolution dynamics."""
        # u represents vertex velocity
        return u
    
    def f_u(self, q):
        """Partial derivative of dynamics wrt control."""
        N = q.shape[0]
        return np.repeat(np.eye(self.q_dim)[None,:,:], N, axis=0)
    
    def L(self, q, u):
        """Running cost: control effort + area constraint."""
        return 0.1 * np.sum(u**2, axis=1) + self.g(q)
    
    def g(self, q):
        """Terminal cost: perimeter + area constraint."""
        perimeter = self.compute_perimeter(q)
        area = self.compute_area(q)
        area_penalty = self.area_weight * (area - self.target_area)**2
        return perimeter + area_penalty
    
    def sample_q(self, num_examples, mode='train'):
        """Sample initial shapes as regular polygons with noise."""
        n_vertices = self.q_dim//2
        angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
        
        # Create regular polygon vertices
        if mode == 'train':
            r = 1.0 + 0.1*np.random.randn(num_examples, 1)
        else:
            r = 1.0 + 0.01*np.random.randn(num_examples, 1)
            
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        
        # Add noise to vertex positions
        if mode == 'train':
            noise_scale = 0.1
        else:
            noise_scale = 0.01
            
        x += noise_scale * np.random.randn(num_examples, n_vertices)
        y += noise_scale * np.random.randn(num_examples, n_vertices)
        
        # Flatten to match q_dim
        q = np.stack([x, y], axis=2).reshape(num_examples, self.q_dim)
        return q
    
    def eval(self, q):
        """Evaluation metric: perimeter + area constraint violation."""
        return self.g(q)

def create_networks(q_dim, z_dim=8):
    """Create neural networks for the shape optimization problem."""
    # Adjoint network
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, 
                  layer_dims=[64, 64], activation='tanh')
    
    # Hamiltonian network
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, 
               layer_dims=[64, 64], activation='tanh')
    
    # Hamiltonian decoder network
    hnet_decoder = Mlp(input_dim=2*q_dim, output_dim=1, 
                       layer_dims=[64, 64], activation='tanh')
    
    # Latent encoder
    z_encoder = Encoder(input_dim=2*q_dim, output_dim=z_dim,
                       share_layer_dims=[64, 32],
                       mean_layer_dims=[16],
                       logvar_layer_dims=[16],
                       activation='tanh')
    
    # Latent decoder
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim,
                    layer_dims=[32, 64], activation='tanh')
    
    return adj_net, hnet, hnet_decoder, z_encoder, z_decoder

def main():
    # Problem setup
    q_dim = 20  # 10 vertices with x,y coordinates
    env = ShapeOptEnv(q_dim=q_dim, u_dim=q_dim)
    
    # Create networks
    adj_net, hnet, hnet_decoder, z_encoder, z_decoder = create_networks(q_dim)
    
    # Training parameters
    num_examples = 1000
    batch_size1 = 32
    batch_size2 = 32
    num_epoch1 = 5 #def 50
    num_epoch2 = 5 #def 30
    num_iter1 = 100  # Increased from default 20
    num_iter2 = 100  # Increased from default 20
    lr1 = 1e-3
    lr2 = 1e-3
    log_interval1 = 10  # Decreased from default 50
    log_interval2 = 10  # Decreased from default 50
    
    # Generate training data
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), 
                            dtype=torch.float)
    
    # Train networks using NeuralPMP
    main_training(env, "shape_opt", q_samples,
                 adj_net, hnet, hnet_decoder, z_decoder, z_encoder,
                 T1=1.0, T2=1.0, control_coef=0.1, dynamic_hidden=False,
                 alpha1=1.0, alpha2=0.1, beta1=1.0, beta2=1.0,
                 num_epoch1=num_epoch1, num_epoch2=num_epoch2,
                 num_iter1=num_iter1, num_iter2=num_iter2,
                 batch_size1=batch_size1, batch_size2=batch_size2,
                 lr1=lr1, lr2=lr2,
                 log_interval1=log_interval1, log_interval2=log_interval2,
                 mode=1)

    # # Visualize results after training
    # print("\nCreating visualization...")
    # anim = visualize_training_results(env, adj_net, hnet, num_shapes=5)
    # save_evolution_video(anim, 'shape_evolution.mp4')
    # print("Visualization saved as 'shape_evolution.mp4'")

if __name__ == "__main__":
    main()