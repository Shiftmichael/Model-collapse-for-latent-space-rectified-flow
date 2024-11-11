import os
import sys
import copy
import time
import random
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
# from torchcfm.models.unet.unet import UNetModelWrapper
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from torchdyn.models import NeuralODE
from torchvision.utils import save_image


def seed_everything(seed=42):
    """
    Seed all random number generators to ensure reproducibility.
    
    :param seed: Integer seed for the random number generators.
    """
    random.seed(seed)           # Python random module
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch
    torch.cuda.manual_seed(seed)  # GPU for PyTorch
    
    # If using multi-GPU, this is recommended by the PyTorch documentation.
    # It helps ensure that each GPU will start from the same random weights after seeding.
    torch.cuda.manual_seed_all(seed)  
    
    # Configure some PyTorch settings for further consistency in results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"All random generators seeded with {seed}")

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

class Config:
    num_channel = 128
    integration_steps = 100
    # integration_method = "dopri5"
    integration_method = "euler"
    step = 400000
    num_gen = 50000
    tol = 1e-5
    batch_size_fid = 1024
    local_rank = 0

config = Config()

def load_checkpoint_and_evaluate(model, checkpoint_path, rank):
    # Step 2: 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))

    # Step 3: 应用检查点到模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['net_model'])
    else:
        model.load_state_dict(checkpoint['net_model'])

    # Step 4: 设置模型为评估模式
    model.eval()


def gen_1_img_ddp(model, device, batch_size_fid):
    with torch.no_grad():
        x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
        if config.integration_method == "euler":
            node = NeuralODE(model, solver="euler")
            t_span = torch.linspace(0, 1, config.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                model, x, t_span, rtol=config.tol, atol=config.tol, method=config.integration_method
            )
    traj = traj[-1, :]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img

class ImageDataset(Dataset):
    def __init__(self, noise, generated, transform=None):
        """Initializes the dataset with noise and generated images and optional transforms."""
        assert len(noise) == len(generated), "Noise and images must have the same length"
        self.noise = noise
        self.generated = generated
        self.transform = transform

    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        noise_img = self.noise[idx]
        generated_img = self.generated[idx]

        if self.transform:
            generated_img = self.transform(generated_img)

        return noise_img, generated_img

# def generate_image_pairs(model, device, rank, total_num_gpus, dataset_size=500000, batch_size=64):
#     model.eval()
#     subset_size = dataset_size // total_num_gpus
#     start_index = rank * subset_size

#     model_ = copy.deepcopy(model)
#     model_ = model_.module.to(device)

#     node_ = NeuralODE(model_.to(device), solver="euler", sensitivity="adjoint")
#     start_time = time.time()
#     all_noise = []
#     all_generated = []

#     with torch.no_grad():
#         for i in range(subset_size // batch_size):
#             noise = torch.randn(batch_size, 3, 32, 32, device=device)
#             traj = node_.trajectory(noise, t_span=torch.linspace(0, 1, 100, device=device))[-1]
#             generated = traj.view(-1, 3, 32, 32).clip(-1, 1)
#             # .div(2).add(0.5)  # Normalize images
#             # test
#             # print(f"noise min: {noise.min()}, max: {noise.max()}")
#             # print(f"generated min: {generated.min()}, max: {generated.max()}")
#             # sys.exit()

#             all_noise.append(noise.cpu())
#             all_generated.append(generated.cpu())

#     all_noise = torch.cat(all_noise)
#     all_generated = torch.cat(all_generated)

#     # transform = transforms.Compose([
#     #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     # ])
#     dataset = ImageDataset(all_noise, all_generated)
#     duration = time.time() - start_time
#     print(f"Generated subset in {duration:.2f} seconds on GPU {rank}")

#     model.train()
#     return dataset

def generate_image_pairs(model, device, rank, total_num_gpus, dataset_size=500000, batch_size=64):
    model.eval()
    subset_size = dataset_size // total_num_gpus

    # 深拷贝模型并移动到指定设备
    model_ = copy.deepcopy(model).to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    start_time = time.time()
    all_noise = []
    all_generated = []

    with torch.no_grad():
        for i in range(subset_size // batch_size):
            noise = torch.randn(batch_size, 3, 32, 32, device=device)
            traj = node_.trajectory(noise, t_span=torch.linspace(0, 1, 100, device=device))[-1]
            generated = traj.view(-1, 3, 32, 32).clamp(-1, 1)

            all_noise.append(noise.cpu())
            all_generated.append(generated.cpu())

    all_noise = torch.cat(all_noise)
    all_generated = torch.cat(all_generated)

    dataset = ImageDataset(all_noise, all_generated)
    duration = time.time() - start_time
    print(f"Generated subset in {duration:.2f} seconds on GPU {rank}")

    model.train()
    return dataset

def generate_real_image_pairs(model, device, rank, total_num_gpus, dataset, vae, method='ode', dataset_size=50000, batch_size=64, sde_noise_std=1e-3, randomize_steps=False, is_latent_data=False, scale_factor=1.0):
    """
    Generates noise-latent pairs using the given dataset by performing inverse sampling
    through a Neural ODE or Neural SDE model.

    Args:
        model (torch.nn.Module): The trained Neural ODE/SDE model.
        device (torch.device): The device to run computations on.
        rank (int): The rank/index of the current GPU (for distributed settings).
        total_num_gpus (int): Total number of GPUs being used.
        dataset (torch.utils.data.Dataset): The dataset to use for generating pairs.
        vae (torch.nn.Module): The VAE encoder model to map images to latent space.
        method (str): 'ode' for deterministic inversion, 'sde' for stochastic inversion.
        dataset_size (int, optional): Total number of samples to generate. Defaults to 50000.
        batch_size (int, optional): Batch size for processing. Defaults to 64.
        sde_noise_std (float, optional): Standard deviation of Gaussian noise for SDE inversion.
        randomize_steps (bool, optional): Whether to randomize integration steps for SDE inversion.
        is_latent_data (bool, optional): Whether the input data is already in latent space.
        scale_factor (float, optional): Scaling factor for latent space.

    Returns:
        ImageDataset: A dataset containing noise-latent pairs.
    """
    assert method in ['ode', 'sde'], "Method must be either 'ode' or 'sde'."

    # Set the model to evaluation mode
    model.eval()
    subset_size = dataset_size // total_num_gpus

    # Deepcopy the model to avoid modifying the original and move it to the specified device
    model_ = copy.deepcopy(model).to(device)

    # Initialize the appropriate node based on the method
    if method == 'ode':
        node = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    elif method == 'sde':
        node = NeuralSDE(model_, solver="euler", sensitivity="adjoint", noise_std=sde_noise_std, randomize_steps=randomize_steps)
    else:
        raise ValueError("Unsupported method. Choose 'ode' or 'sde'.")
    
    start_time = time.time()
    all_noise = []
    all_latents = []

    # Create a DataLoader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        for batch in dataloader:
            # Assume dataset returns (x, y), ignore y if present
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            x = x.to(device)

            # Encode images to latent space using VAE encoder
            if is_latent_data:
                # z_0 = x * scale_factor
                z_0 = x
            else:
                z_0 = vae.encode(x).latent_dist.sample().mul_(scale_factor)

            # Perform inverse trajectory from t=1 to t=0
            if method == 'ode':
                # For ODE method, take the last state
                traj = node.trajectory(z_0, t_span=torch.linspace(1, 0, 100, device=device))[-1]
                noise = traj.view(z_0.size()).to(device)
            elif method == 'sde':
                # For SDE method, take the entire trajectory and use the last state
                traj = node.trajectory(z_0, t_span=torch.linspace(1, 0, 100, device=device))
                noise = traj[-1].view(z_0.size()).to(device)

            # Append to the lists
            all_noise.append(noise.cpu())
            all_latents.append(z_0.cpu())

    # Concatenate all batches and trim to the exact subset size
    all_noise = torch.cat(all_noise)[:subset_size]
    all_latents = torch.cat(all_latents)[:subset_size]

    # Create the dataset with noise-latent pairs
    dataset = ImageDataset(all_noise, all_latents)

    duration = time.time() - start_time
    print(f"Generated subset in {duration:.2f} seconds on GPU {rank} using method '{method}'")

    # Set the model back to training mode
    model.train()

    return dataset


# def generate_real_image_pairs(model, device, rank, total_num_gpus, dataset, vae, method='ode', dataset_size=50000, batch_size=64, cifar_root='./data', sde_noise_std=1e-3, randomize_steps=False):
#     assert method in ['ode', 'sde'], "Method must be either 'ode' or 'sde'."

#     model.eval()
#     subset_size = dataset_size // total_num_gpus

#     # 深拷贝模型并移动到指定设备
#     model_ = copy.deepcopy(model).to(device)

#     if method == 'ode':
#         node = NeuralODE(model_, solver="euler", sensitivity="adjoint")
#     elif method == 'sde':
#         node = NeuralSDE(model_, solver="euler", sensitivity="adjoint", noise_std=sde_noise_std, randomize_steps=randomize_steps)

#     start_time = time.time()
#     all_noise = []
#     all_images = []

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     cifar_dataset = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)
#     cifar_length = len(cifar_dataset)
#     if subset_size > cifar_length:
#         repeat_times = (subset_size + cifar_length - 1) // cifar_length
#         indices = list(range(cifar_length)) * repeat_times
#     else:
#         indices = list(range(cifar_length))

#     start_index = rank * subset_size
#     subset_indices = indices[start_index:start_index + subset_size]
#     subset = Subset(cifar_dataset, subset_indices)

#     dataloader = DataLoader(
#         subset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     with torch.no_grad():
#         for batch in dataloader:
#             images, _ = batch
#             images = images.to(device)

#             if method == 'ode':
#                 traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))[-1]
#                 noise = traj.view(-1, 3, 32, 32)
#             elif method == 'sde':
#                 traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))
#                 noise = traj[-1].view(-1, 3, 32, 32)

#             all_noise.append(noise.cpu())
#             all_images.append(images.cpu())

#     all_noise = torch.cat(all_noise)[:subset_size]
#     all_images = torch.cat(all_images)[:subset_size]

#     dataset = ImageDataset(all_noise, all_images)

#     duration = time.time() - start_time
#     print(f"Generated subset in {duration:.2f} seconds on GPU {rank} using method '{method}'")

#     model.train()

#     return dataset


# def generate_real_image_pairs(model, device, rank, total_num_gpus, method='ode', dataset_size=50000, batch_size=64, cifar_root='./data', sde_noise_std=1e-3, randomize_steps=False):
#     """
#     Generates noise-image pairs using the CIFAR-10 dataset by performing inverse sampling
#     through a Neural ODE or Neural SDE model.

#     Args:
#         model (torch.nn.Module): The trained Neural ODE/SDE model.
#         device (torch.device): The device to run computations on.
#         rank (int): The rank/index of the current GPU (for distributed settings).
#         total_num_gpus (int): Total number of GPUs being used.
#         method (str): 'ode' for deterministic inversion, 'sde' for stochastic inversion.
#         dataset_size (int, optional): Total number of samples to generate. Defaults to 50000.
#         batch_size (int, optional): Batch size for processing. Defaults to 64.
#         cifar_root (str, optional): Root directory for CIFAR-10 data. Defaults to './data'.
#         sde_noise_std (float, optional): Standard deviation of Gaussian noise for SDE inversion.
#         randomize_steps (bool, optional): Whether to randomize integration steps for SDE inversion.

#     Returns:
#         ImageDataset: A dataset containing noise-image pairs.
#     """
#     assert method in ['ode', 'sde'], "Method must be either 'ode' or 'sde'."

#     # Set the model to evaluation mode
#     model.eval()

#     # Determine the subset size for each GPU
#     subset_size = dataset_size // total_num_gpus
#     start_index = rank * subset_size

#     # Deepcopy the model to avoid modifying the original and move it to the specified device
#     model_ = copy.deepcopy(model)
#     if hasattr(model_, 'module'):
#         model_ = model_.module.to(device)
#     else:
#         model_ = model_.to(device)

#     # Initialize the appropriate node based on the method
#     if method == 'ode':
#         node = NeuralODE(model_, solver="euler", sensitivity="adjoint")
#     elif method == 'sde':
#         node = NeuralSDE(model_, solver="euler", sensitivity="adjoint", noise_std=sde_noise_std, randomize_steps=randomize_steps)
#     else:
#         raise ValueError("Unsupported method. Choose 'ode' or 'sde'.")

#     start_time = time.time()
#     all_noise = []
#     all_images = []

#     # Define the transformation for CIFAR-10 images
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
#     ])

#     # Load the CIFAR-10 dataset
#     cifar_dataset = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)

#     # Handle cases where dataset_size exceeds CIFAR-10 size by repeating the dataset
#     cifar_length = len(cifar_dataset)
#     if subset_size > cifar_length:
#         repeat_times = (subset_size + cifar_length - 1) // cifar_length  # Ceiling division
#         indices = list(range(cifar_length)) * repeat_times
#     else:
#         indices = list(range(cifar_length))

#     # Select the subset for the current GPU
#     subset_indices = indices[start_index:start_index + subset_size]
#     subset = Subset(cifar_dataset, subset_indices)

#     # Create a DataLoader for the subset
#     dataloader = DataLoader(
#         subset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     # Perform inverse sampling without tracking gradients
#     with torch.no_grad():
#         for batch in dataloader:
#             images, _ = batch  # Ignore labels
#             images = images.to(device)

#             if method == 'ode':
#                 # Perform inverse trajectory from t=1 to t=0
#                 traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))[-1]
#                 noise = traj.view(-1, 3, 32, 32)  # do not Ensure noise is within [-1, 1]
#             elif method == 'sde':
#                 # Perform inverse trajectory with stochasticity from t=1 to t=0
#                 traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))
#                 # traj is a list of states; take the last state as noise
#                 noise = traj[-1].view(-1, 3, 32, 32)
#             else:
#                 raise ValueError("Unsupported method. Choose 'ode' or 'sde'.")

#             # test
#             # print(f"noise min: {noise.min()}, max: {noise.max()}")
#             # print(f"images min: {images.min()}, max: {images.max()}")
#             # sys.exit()
#             # test
#             # print(noise.shape)
#             # sys.exit()

#             # Append to the lists
#             all_noise.append(noise.cpu())
#             all_images.append(images.cpu())

#     # Concatenate all batches and trim to the exact subset size
#     all_noise = torch.cat(all_noise)[:subset_size]
#     all_images = torch.cat(all_images)[:subset_size]

#     # test 
#     # print(f"noise: {all_noise.shape}")
#     # print(f"images: {all_images.shape}")
#     # sys.exit()

#     # Create the dataset with noise-image pairs
#     dataset = ImageDataset(all_noise, all_images)

#     duration = time.time() - start_time
#     print(f"Generated subset in {duration:.2f} seconds on GPU {rank} using method '{method}'")

#     # Set the model back to training mode
#     model.train()

#     return dataset

class NeuralSDE(nn.Module):
    def __init__(self, model, solver='euler', sensitivity='adjoint', noise_std=1e-3, randomize_steps=False):
        super(NeuralSDE, self).__init__()
        self.model = model
        self.solver = solver
        self.sensitivity = sensitivity
        self.noise_std = noise_std
        self.randomize_steps = randomize_steps

    def trajectory(self, x, t_span):
        """
        Computes the trajectory using a stochastic differential equation.

        Args:
            x (torch.Tensor): Initial state.
            t_span (torch.Tensor): Tensor of timesteps.

        Returns:
            list of torch.Tensor: Trajectory states at each timestep.
        """
        traj = [x]
        dt = (t_span[-1] - t_span[0]) / (len(t_span) - 1)

        current_x = x
        for i in range(1, len(t_span)):
            t = t_span[i-1]
            if self.randomize_steps:
                # Randomize step size within a small range, e.g., ±10%
                step_variation = 0.1 * dt
                dt_step = dt + torch.randn(1).item() * step_variation
            else:
                dt_step = dt

            # test
            # print(current_x.shape)
            # sys.exit()
            # Compute deterministic part using the model
            f = self.model(t, current_x)
            deterministic = current_x + f * dt_step
            # print("down")
            # sys.exit()

            # Add stochastic noise
            noise = torch.randn_like(current_x) * self.noise_std * torch.sqrt(dt_step.detach())

            current_x = deterministic + noise
            traj.append(current_x)

        return traj
    
def mix_and_shuffle_batches(batch_generated, batch_real, real_ratio=0.5):
    """
    Mix the data of two batches and shuffle them by the specified ratio while keeping the pairing relationship between z_0 and z_1.

    Args:
        batch_generated (tuple): batches of generated data, including (z0_gen, z1_gen).
        batch_real (tuple): batches of real data, including (z0_real, z1_real).
        real_ratio (float): the ratio of real data in the mixed batch, in the range [0, 1].

    Returns:
        tuple: mixed and shuffled (mixed_z0, mixed_z1).
    """
    z0_gen, z1_gen = batch_generated
    z0_real, z1_real = batch_real

    batch_size_gen = z0_gen.size(0)
    batch_size_real = z0_real.size(0)

    # 确保 real_ratio 在合理范围内
    real_ratio = max(0.0, min(real_ratio, 1.0))

    # 计算从生成数据和真实数据中各抽取的样本数量
    num_real = int(real_ratio * batch_size_gen)
    num_gen = batch_size_gen - num_real

    # 如果真实数据批次不足，则调整数量
    num_real = min(num_real, batch_size_real)
    num_gen = min(num_gen, batch_size_gen)

    # 随机选择样本索引
    if num_real > 0:
        perm_real = torch.randperm(batch_size_real)[:num_real]
        selected_z0_real = z0_real[perm_real]
        selected_z1_real = z1_real[perm_real]
    else:
        selected_z0_real = torch.empty(0, *z0_real.shape[1:], device=z0_real.device)
        selected_z1_real = torch.empty(0, *z1_real.shape[1:], device=z1_real.device)

    if num_gen > 0:
        perm_gen = torch.randperm(batch_size_gen)[:num_gen]
        selected_z0_gen = z0_gen[perm_gen]
        selected_z1_gen = z1_gen[perm_gen]
    else:
        selected_z0_gen = torch.empty(0, *z0_gen.shape[1:], device=z0_gen.device)
        selected_z1_gen = torch.empty(0, *z1_gen.shape[1:], device=z1_gen.device)

    # 合并选中的生成数据和真实数据
    mixed_z0 = torch.cat([selected_z0_gen, selected_z0_real], dim=0)
    mixed_z1 = torch.cat([selected_z1_gen, selected_z1_real], dim=0)

    # 生成随机排列索引
    if mixed_z0.size(0) > 0:
        perm = torch.randperm(mixed_z0.size(0))
        mixed_z0 = mixed_z0[perm]
        mixed_z1 = mixed_z1[perm]
    else:
        # 如果没有数据，返回空张量
        mixed_z0 = torch.empty(0, *z0_gen.shape[1:], device=z0_gen.device)
        mixed_z1 = torch.empty(0, *z1_gen.shape[1:], device=z1_gen.device)

    return mixed_z0, mixed_z1


# def mix_and_shuffle_batches(batch_generated, batch_real, real_ratio=0.5):
#     """
#     混合两个批次的数据并按指定比例乱序，同时保持x0和x1的配对关系。

#     Args:
#         batch_generated (tuple): 生成数据的批次，包含 (x0, x1)。
#         batch_real (tuple): 真实数据的批次，包含 (real_x0, real_x1)。
#         real_ratio (float): 真实数据在混合批次中的比例，范围 [0, 1]。

#     Returns:
#         tuple: 混合并乱序后的 (mixed_x0, mixed_x1)。
#     """
#     x0_gen, x1_gen = batch_generated
#     x0_real, x1_real = batch_real

#     # test
#     # print(x0_gen.shape)
#     # print(x1_gen.shape)
#     # print(x1_real.shape)
#     # print(x1_real.shape)
#     # sys.exit()

#     batch_size_gen = x0_gen.size(0)
#     batch_size_real = x0_real.size(0)

#     #test 
#     # print(f"{batch_size_gen == x1_gen.size(0)}")
#     # print(f"{batch_size_real == x1_real.size(0)}")

#     # 确保real_ratio在合理范围内
#     real_ratio = max(0.0, min(real_ratio, 1.0))

#     # 计算从生成数据和真实数据中各抽取的样本数量
#     num_real = int(real_ratio * batch_size_gen)
#     num_gen = batch_size_gen - num_real

#     # 如果真实数据批次不足，则调整数量
#     num_real = min(num_real, batch_size_real)
#     num_gen = min(num_gen, batch_size_gen)

#     # 随机选择样本索引
#     if num_real > 0:
#         perm_real = torch.randperm(batch_size_real)[:num_real]
#         selected_x0_real = x0_real[perm_real]
#         selected_x1_real = x1_real[perm_real]
#     else:
#         selected_x0_real = torch.empty(0, *x0_real.shape[1:], device=x0_real.device)
#         selected_x1_real = torch.empty(0, *x1_real.shape[1:], device=x1_real.device)

#     if num_gen > 0:
#         perm_gen = torch.randperm(batch_size_gen)[:num_gen]
#         selected_x0_gen = x0_gen[perm_gen]
#         selected_x1_gen = x1_gen[perm_gen]
#     else:
#         selected_x0_gen = torch.empty(0, *x0_gen.shape[1:], device=x0_gen.device)
#         selected_x1_gen = torch.empty(0, *x1_gen.shape[1:], device=x1_gen.device)

#     # 合并选中的生成数据和真实数据
#     mixed_x0 = torch.cat([selected_x0_gen, selected_x0_real], dim=0)
#     mixed_x1 = torch.cat([selected_x1_gen, selected_x1_real], dim=0)

#     # 生成随机排列索引
#     if mixed_x0.size(0) > 0:
#         perm = torch.randperm(mixed_x0.size(0))
#         mixed_x0 = mixed_x0[perm]
#         mixed_x1 = mixed_x1[perm]
#     else:
#         # 如果没有数据，返回空张量
#         mixed_x0 = torch.empty(0, *x0_gen.shape[1:], device=x0_gen.device)
#         mixed_x1 = torch.empty(0, *x1_gen.shape[1:], device=x1_gen.device)

#     return mixed_x0, mixed_x1

def batch_generate_image_pairs(model, device, batch_size=64):
    """
    Generates a batch of noise and generated image pairs.

    Args:
        model (torch.nn.Module): The trained model.
        device (torch.device): The computation device.
        batch_size (int): Number of samples to generate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of noise images and generated images.
    """
    model.eval()
    model_ = copy.deepcopy(model)
    model_ = model_.module.to(device)
    
    
    node = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    
    with torch.no_grad():
        # Generate 'batch_size' number of noise images
        noise = torch.randn(batch_size, 3, 32, 32, device=device)
        # Generate images through the model
        traj = node.trajectory(noise, t_span=torch.linspace(0, 1, 100, device=device))[-1]
        generated = traj.view(-1, 3, 32, 32).clamp(-1, 1)
    
    model.train()
    return noise.cpu(), generated.cpu()

def batch_generate_real_image_pairs(model, device, dataset, vae, method='ode', batch_size=64, cifar_root='./data', sde_noise_std=1e-3, randomize_steps=False):
    """
    Generates a batch of noise-image pairs by inversely feeding real images into the model.

    Args:
        model (torch.nn.Module): The trained model.
        device (torch.device): The computation device.
        method (str): 'ode' or 'sde' inversion method.
        batch_size (int): Number of samples to generate.
        cifar_root (str): Root directory for CIFAR-10 data.
        sde_noise_std (float): Noise standard deviation for SDE inversion.
        randomize_steps (bool): Whether to randomize integration steps in SDE inversion.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of noise images and real images.
    """
    assert method in ['ode', 'sde'], "Method must be 'ode' or 'sde'."

    model.eval()
    model_ = copy.deepcopy(model).to(device)
    model_ = model_.module.to(device)

    if method == 'ode':
        node = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    elif method == 'sde':
        node = NeuralSDE(model_, solver="euler", sensitivity="adjoint", noise_std=sde_noise_std, randomize_steps=randomize_steps)
    else:
        raise ValueError("Unsupported method. Choose 'ode' or 'sde'.")

    # Define transformations for CIFAR-10 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load the CIFAR-10 dataset
    cifar_dataset = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)
    cifar_length = len(cifar_dataset)
    # Randomly select 'batch_size' images
    indices = torch.randperm(cifar_length)[:batch_size]
    subset = Subset(cifar_dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch  # Ignore labels
            images = images.to(device)

            if method == 'ode':
                # Integrate backward from t=1 to t=0
                traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))[-1]
                noise = traj.view(-1, 3, 32, 32)
            elif method == 'sde':
                # Integrate backward with stochasticity
                traj = node.trajectory(images, t_span=torch.linspace(1, 0, 100, device=device))
                noise = traj[-1].view(-1, 3, 32, 32)
            else:
                raise ValueError("Unsupported method. Choose 'ode' or 'sde'.")
            break  # Only need one batch

    model.train()
    return noise.cpu(), images.cpu()



# class ImageDataset(Dataset):
#     def __init__(self, noise, generated):
#         """Initializes the dataset with noise and generated images."""
#         self.noise = noise
#         self.generated = generated

#     def __len__(self):
#         return self.noise.shape[0]

#     def __getitem__(self, idx):
#         return self.noise[idx], self.generated[idx]

# def generate_image_pairs(model, device, rank, total_num_gpus, dataset_size=500000, batch_size=64):
#     """Generates image pairs and saves them individually, only a subset per GPU."""
#     model.eval()
#     subset_size = dataset_size // total_num_gpus  # 每个GPU生成的图像对数量
#     start_index = rank * subset_size  # 每个GPU的起始索引

#     model_ = copy.deepcopy(model)
#     model_ = model_.module.to(device)

#     node_ = NeuralODE(model_.to(device), solver="euler", sensitivity="adjoint")
#     # os.makedirs(f"data/{rank}", exist_ok=True)

#     start_time = time.time()
#     all_noise = []
#     all_generated = []

#     with torch.no_grad():
#         for i in range(subset_size // batch_size):
#             noise = torch.randn(batch_size, 3, 32, 32, device=device)
#             traj = node_.trajectory(noise, t_span=torch.linspace(0, 1, 100, device=device))[-1]
#             generated = traj.view(-1, 3, 32, 32).clip(-1, 1).div(2).add(0.5)  # Normalize images

#             # Append batches for DataLoader creation
#             all_noise.append(noise.cpu())
#             all_generated.append(generated.cpu())

#             # Save images
#             # for j in range(batch_size):
#             #     save_image(noise[j], f"data/{rank}/noise_{start_index + i * batch_size + j}.png")
#             #     save_image(generated[j], f"data/{rank}/generated_{start_index + i * batch_size + j}.png")

#     # Concatenate and create dataset
#     all_noise = torch.cat(all_noise)
#     all_generated = torch.cat(all_generated)
#     dataset = ImageDataset(all_noise, all_generated)
#     # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     duration = time.time() - start_time
#     print(f"Generated subset in {duration:.2f} seconds on GPU {rank}")

#     model.train()
#     return dataset

def encode_dataset_to_latent(dataset, vae, device, batch_size=64, num_workers=4, is_latent_data=False, scale_factor=1.0):
    """
    Encodes all images in the dataset into the latent space using the VAE encoder
    and reconstructs a new dataset containing latent representations.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset with images.
        vae (torch.nn.Module): The VAE encoder model to map images to latent space.
        device (torch.device): The device to run computations on.
        batch_size (int, optional): Batch size for processing. Defaults to 64.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        is_latent_data (bool, optional): Whether the input data is already in latent space.
        scale_factor (float, optional): Scaling factor for latent space.

    Returns:
        latent_dataset (torch.utils.data.TensorDataset): A dataset containing latent representations and labels (if any).
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Ensure the VAE is in evaluation mode
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Create a DataLoader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Assume dataset returns (x, y); adjust if dataset returns differently
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                all_labels.append(y)
            else:
                x = batch
                y = None

            # Move input to the specified device
            x = x.to(device)

            # Encode images to latent space using VAE encoder
            if is_latent_data:
                z = x * scale_factor
            else:
                z = vae.encode(x).latent_dist.sample().mul_(scale_factor)

            # Move latent representations to CPU and store
            all_latents.append(z.cpu())

    # Concatenate all latent tensors
    all_latents = torch.cat(all_latents)

    if all_labels:
        # Concatenate all labels if available
        all_labels = torch.cat(all_labels)
        latent_dataset = TensorDataset(all_latents, all_labels)
    else:
        latent_dataset = TensorDataset(all_latents)

    return latent_dataset


