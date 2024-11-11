import os
import sys
import copy

import matplotlib.pyplot as plt
import torch
# from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper

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

def calculate_fid_ddp(model_, device, batch_size_fid, num_gen, rank):
    if rank == 0:
        model_.eval()
        model = copy.deepcopy(model_)
        model = model.module.to(device)
        score = fid.compute_fid(
            gen=lambda x: gen_1_img_ddp(model, device, batch_size_fid),
            dataset_name="cifar10",
            batch_size=batch_size_fid,
            dataset_res=32,
            num_gen=num_gen,
            dataset_split="train",
            mode="legacy_tensorflow",
        )
        model_.train()
        return score
    return None

def gen_1_img_ddp(model, device, batch_size_fid):
    torch.manual_seed(42) 
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


# def calculate_fid(model, device, batch_size_fid, num_gen):
#     print("Start computing FID")
#     score = fid.compute_fid(
#         gen=lambda x: gen_1_img(model, device, batch_size_fid),
#         dataset_name="cifar10",
#         batch_size=batch_size_fid,
#         dataset_res=32,
#         num_gen=num_gen,
#         dataset_split="train",
#         mode="legacy_tensorflow",
#     )
#     print("FID has been computed")
#     print("FID: ", score)
#     return score

# def gen_1_img(model, device, batch_size_fid):
#     with torch.no_grad():
#         x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
#         print("Use method: ", FLAGS.integration_method)
#         t_span = torch.linspace(0, 1, 2, device=device)
#         traj = odeint(
#             model, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
#         )
#     traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
#     img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
#     return img

# def calculate_fid_ddp(model, device, batch_size_fid, num_gen, rank):
#     if rank == 0:
#         device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#         score = fid.compute_fid(
#             gen=lambda x: gen_1_img_ddp(model.module, device, batch_size_fid),  # 注意使用 model.module 来访
#             dataset_name="cifar10",
#             batch_size=batch_size_fid,
#             dataset_res=32,
#             num_gen=num_gen,
#             dataset_split="train",
#             mode="legacy_tensorflow",
#         )
#         return score
#     return None 


# def gen_1_img_ddp(model, device, batch_size_fid):
#     torch.manual_seed(42)  # 确保所有进程生成相同的随机数
#     with torch.no_grad():
#         x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
#         if FLAGS.integration_method == "euler":
#             t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
#             traj = model.module.trajectory(x, t_span=t_span)
#         else:
#             t_span = torch.linspace(0, 1, 2, device=device)
#             traj = odeint(
#                 model.module, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
#             )
#     traj = traj[-1, :]
#     img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
#     return img
