# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import math
import shutil
from functools import partial
from time import time

import torch
from omegaconf import OmegaConf

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets_prep import get_dataset
from EMA import EMA
from models import create_network
from torchdiffeq import odeint_adjoint as odeint
from utils_reflow.utils import generate_image_pairs, generate_real_image_pairs, mix_and_shuffle_batches, encode_dataset_to_latent, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb


def sample_from_model(model, x_0):
    t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image

def reverse_sample_from_model(model, x_1):
    t = torch.tensor([0.0, 1.0], dtype=x_1.dtype, device="cuda")
    fake_noise = odeint(model, x_1, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_noise


# %%
def train(args):
    from diffusers.models import AutoencoderKL

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    batch_size = args.batch_size

    for reflow_step in range(args.reflow_times):
        if accelerator.is_main_process:
            print(f"reflow_step: {reflow_step}", f"model type: {args.model_type}", f"lr: {args.lr}", f"real ratio:{args.real_ratio}")

        checkpoint_path = "results/{args.dataset}/RCA_{args.real_ratio}/" + f"reflow_{reflow_step}_8.pt"
        checkpoint = torch.load(checkpoint_path,  map_location=device)
        print("Load, dowm")
        # dataset_real = get_dataset(args)
        # data_loader = torch.utils.data.DataLoader(
        #     dataset_real,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     pin_memory=True,
        #     drop_last=True,
        # )

        model = create_network(args).to(device, dtype=dtype)
        if args.use_grad_checkpointing and "DiT" in args.model_type:
            model.set_gradient_checkpointing()

        model.load_state_dict(checkpoint)

        first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
        first_stage_model = first_stage_model.eval()
        first_stage_model.train = False
        for param in first_stage_model.parameters():
            param.requires_grad = False

        rank = accelerator.process_index
        total_num_gpus = accelerator.num_processes
        is_latent_data = True if "latent" in args.dataset else False  
        scale_factor = args.scale_factor

        dataset_real = get_dataset(args)

        dataset_real_latent = encode_dataset_to_latent(
            dataset=dataset_real,
            vae=first_stage_model,  # VAE encoder
            device=device,
            batch_size=batch_size,
            num_workers=4,
            is_latent_data=is_latent_data,
            scale_factor=args.scale_factor
        )

        dataset_syn = generate_image_pairs(
            model, 
            device, 
            rank,
            total_num_gpus, 
            dataset_size=500000, 
            batch_size=1024
        )

        generated_data_loader = DataLoader(
            dataset_syn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # is_latent_data = True if "latent" in args.dataset else False
        # scale_factor = args.scale_factor

        accelerator.print("AutoKL size: {:.3f}MB".format(get_weight(first_stage_model)))
        accelerator.print("FM size: {:.3f}MB".format(get_weight(model)))

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

        if args.use_ema:
            optimizer = EMA(optimizer, ema_decay=args.ema_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

        generated_data_loader, model, optimizer, scheduler = accelerator.prepare(
            generated_data_loader, model, optimizer, scheduler
        )

        gen_data_iter = iter(generated_data_loader)

        # data_loader, model, optimizer, scheduler = accelerator.prepare(data_loader, model, optimizer, scheduler)

        exp = args.exp
        parent_dir = f"./results/{args.dataset}/RCA_{args.real_ratio}"
        exp_path = os.path.join(parent_dir, exp)
        if accelerator.is_main_process:
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                config_dict = vars(args)
                OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
        accelerator.print("Exp path:", exp_path)

        if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
            checkpoint_file = os.path.join(exp_path, "content.pth")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            init_epoch = checkpoint["epoch"]
            epoch = init_epoch
            model.load_state_dict(checkpoint["model_dict"])
            # load G
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            global_step = checkpoint["global_step"]

            accelerator.print("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
            del checkpoint

        elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
            checkpoint_file = os.path.join(exp_path, args.model_ckpt)
            checkpoint = torch.load(checkpoint_file, map_location=device)
            epoch = int(args.model_ckpt.split("_")[-1][:-4])
            init_epoch = epoch
            model.load_state_dict(checkpoint)
            global_step = 0

            accelerator.print("=> loaded checkpoint (epoch {})".format(epoch))
            del checkpoint
        else:
            global_step, epoch, init_epoch = 0, 0, 0

        use_label = True if "imagenet" in args.dataset else False
        is_latent_data = True if "latent" in args.dataset else False
        log_steps = 0
        real_ratio = args.real_ratio
        sigma = 0.0
        FM = ConditionalFlowMatcher(sigma=sigma)
        start_time = time()

        if accelerator.is_main_process:
            steps_per_epoch = math.ceil(len(generated_data_loader.dataset) / (batch_size * (total_num_gpus)))
            print(f"reflow times: {reflow_step}, steps_per_epoch: {steps_per_epoch}, num_epochs: {args.num_epoch}")

        # for epoch in range(init_epoch, args.num_epoch + 1):
        #     for iteration, (x, y) in enumerate(data_loader):
        #         x_0 = x.to(device, dtype=dtype, non_blocking=True)
        #         y = None if not use_label else y.to(device, non_blocking=True)
        for epoch in range(init_epoch, args.num_epoch + 1):
            # Generate the real image pairs
            if epoch % args.renew_alpha == 0:
                real_data = generate_real_image_pairs(
                    model=model,
                    device=device,
                    rank=accelerator.process_index,
                    total_num_gpus=accelerator.num_processes,
                    dataset=dataset_real_latent,  # specified dataset
                    vae=first_stage_model,  # VAE encoder
                    method='ode',  # or 'sde' depending on your needs
                    dataset_size=len(dataset_real_latent),
                    batch_size=1024,
                    sde_noise_std=1e-3,
                    randomize_steps=True,
                    is_latent_data=is_latent_data,
                    scale_factor=scale_factor
                )

                real_data_loader = DataLoader(
                    real_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                )
            
                # Prepare the new real_data_loader
                real_data_loader = accelerator.prepare_data_loader(real_data_loader)

                # Reset the data iterator
                real_data_iter = iter(real_data_loader)

            for iteration in range(len(generated_data_loader)):
                y = None if not use_label else y.to(device, non_blocking=True)
                try:
                    batch_generated = next(gen_data_iter)
                except StopIteration:
                    gen_data_iter = iter(generated_data_loader)
                    batch_generated = next(gen_data_iter)

                try:
                    batch_real = next(real_data_iter)
                except StopIteration:
                    real_data_iter = iter(real_data_loader)
                    batch_real = next(real_data_iter)

                mixed_x0, mixed_x1 = mix_and_shuffle_batches(batch_generated, batch_real, real_ratio=real_ratio)
                x0 = mixed_x0.to(device, dtype=dtype, non_blocking=True)
                x1 = mixed_x1.to(device, dtype=dtype, non_blocking=True)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

                model.zero_grad()
                
                # estimate velocity
                v = model(t.squeeze(), xt, y)
                loss = F.mse_loss(v, ut)
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                log_steps += 1
                if iteration % 100 == 0:
                    if accelerator.is_main_process:
                        # Measure training speed:
                        end_time = time()
                        steps_per_sec = log_steps / (end_time - start_time)
                        accelerator.print(
                            "epoch {} iteration{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                                epoch, iteration, loss.item(), steps_per_sec
                            )
                        )
                        # Reset monitoring variables:
                        log_steps = 0
                        start_time = time()

            if not args.no_lr_decay:
                scheduler.step()

            if accelerator.is_main_process:
                if epoch % args.plot_every == 0:
                    with torch.no_grad():
                        rand = torch.randn_like(x1)[:4]
                        if y is not None:
                            y = y[:4]
                        sample_model = partial(model, y=y)
                        # sample_func = lambda t, x: model(t, x, y=y)
                        fake_sample = sample_from_model(sample_model, rand)[-1]
                        fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                    torchvision.utils.save_image(
                        fake_image,
                        os.path.join(exp_path, "image_epoch_{}.png".format(epoch)),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    accelerator.print("Finish sampling")

                if args.save_content:
                    if epoch % args.save_content_every == 0:
                        accelerator.print("Saving content.")
                        content = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "args": args,
                            "model_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }

                        torch.save(content, os.path.join(exp_path, "content.pth"))

                if epoch % args.save_ckpt_every == 0:
                    if args.use_ema:
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                    torch.save(
                        model.state_dict(),
                        os.path.join(exp_path, "model_{}.pth".format(epoch)),
                    )
                    if args.use_ema:
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")

    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        help="model_type",
        choices=[
            "adm",
            "ncsn++",
            "ddpm++",
            "DiT-B/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
        ],
    )
    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )

    # Original ADM
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--use_origin_adm", action="store_true")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=400)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")

    # RCA
    parser.add_argument("--reflow_times", type=int, default=5, help="reflow times")
    parser.add_argument("--real_ratio", type=float, default=0.5, help="real ratio")
    parser.add_argument("--total_gpu", type=int, default=2, help="real ratio")
    parser.add_argument("--renew_alpha", type=int, default=2, help="real ratio")



    args = parser.parse_args()
    train(args)
