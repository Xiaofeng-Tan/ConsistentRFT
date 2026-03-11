# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.
from typing import Dict, Any
import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
import swanlab
import random
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_training_w_o_optimizer,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
from contextlib import contextmanager
from safetensors.torch import save_file
from sklearn.cluster import KMeans
import copy

# def load_hpsv2(args):
    

class FSDP_EMA:
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0, 
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0, 
                options=restore_options
            )

def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    if rank == 0 and ema_handler is not None:
        ema_checkpoint_path = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_checkpoint_path, exist_ok=True)
        weight_path = os.path.join(ema_checkpoint_path ,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        # save dict as json
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        #torch.save(ema_handler.ema_state_dict_rank0, os.path.join(ema_checkpoint_path, "ema_model.pt"))
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
        stop_step,
    ):
    if grpo_sample:
        if args.consist:
            all_predict_ori = []
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            if i == stop_step:
                break
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps/1000,
                    guidance=torch.tensor(
                        [3.5],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
            if args.consist:
                all_predict_ori.append(pred_original)
        if args.consist:
            all_predict_ori.append(all_predict_ori[-1])
            all_predict_ori = torch.stack(all_predict_ori, dim=1)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        if args.consist:
            return z, latents, all_latents, all_log_probs, all_predict_ori
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids.squeeze(0),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    if args.consist:
        return log_prob, pred_original
    return log_prob

def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    clip_model = None,
    preprocess_dgn5b = None,
    clip_tokenizer = None,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []
    all_clip_score = []
    all_image_ids = []
    reward_list = []
    if args.consist:
        all_predict_ori = []
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            sample_ret = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample,
                stop_step = int(args.diversity_step) if args.diversity else int(sample_steps * args.skip_rate)
            )
            if args.consist:
                z, latents, batch_latents, batch_log_probs, batch_predict_ori = sample_ret
            else:
                z, latents, batch_latents, batch_log_probs = sample_ret
            #print("z.shape", z.shape)
        
        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        if args.consist:
            all_predict_ori.append(batch_predict_ori)
        vae.enable_tiling()
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                #print("1", latents.shape)
                latents = unpack_latents(latents, h, w, 8)
                #print("2", latents.shape)
                #exit()
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(
                image)
        decoded_image[0].save(f"./{args.save_image_path}/rough_flux_{rank}_{index}.png")
        # print(latents.shape)
        # exit()
        if args.use_hpsv2:
            with torch.no_grad():
                image_path = decoded_image[0]
                image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    outputs = reward_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                    if args.use_hpsv2_clip:
                        image = preprocess_dgn5b(image_path.convert("RGB")).unsqueeze(0).to(device=device, non_blocking=True)
                        text = clip_tokenizer([batch_caption[0]], context_length=clip_model.context_length).to(device=device, non_blocking=True)
                        with torch.no_grad():
                            with torch.amp.autocast('cuda'):
                                    image_features = clip_model.encode_image(image)
                                    text_features = clip_model.encode_text(text)
                                    image_features = F.normalize(image_features, dim=-1)
                                    text_features = F.normalize(text_features, dim=-1)
                                    clip_score = image_features @ text_features.T 
                            #clip_reward.append(clip_score.squeeze(0))
                            all_clip_score.append(clip_score)
                all_rewards.append(hps_score)
        
        if args.use_pickscore:
            def calc_probs(processor, model, prompt, images, device):
                # preprocess
                image_inputs = processor(
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                text_inputs = processor(
                    text=prompt,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    # embed
                    image_embs = model.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                    text_embs = model.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                
                    # score
                    scores = (text_embs @ image_embs.T)[0]
                
                return scores
            pil_images = [Image.open(f"./images/flux_{rank}_{index}.png")]
            score = calc_probs(tokenizer, reward_model, caption, pil_images, device)
            all_rewards.append(score)

    all_rewards = torch.cat(all_rewards, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)
    if args.consist:
        all_predict_ori = torch.cat(all_predict_ori, dim=0)
    #print(all_latents.shape)
    #exit()
    if args.use_hpsv2_clip:
        all_clip_score = torch.cat(all_clip_score, dim=0)
        return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_clip_score
    else:
        if args.consist:
            return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_predict_ori
        else:
            return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def vis_image_only(vae, index, ori, latents):
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    rank = int(os.environ["RANK"])
    w, h = 720, 720
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = unpack_latents(latents, h, w, 8)
            latents = (latents / 0.3611) + 0.1159
            image = vae.decode(latents, return_dict=False)[0]
            decoded_image = image_processor.postprocess(
            image)
    decoded_image[0].save(f"./{args.save_image_path}/latents_{rank}_{index}_{ori}.png")

def flow_forward(batch_img_1_embed, sigma_schedule, progress_bar, vae):
    noise = torch.randn_like(batch_img_1_embed)
    traj = [noise]
    for i in progress_bar:
        sigma_t = sigma_schedule[i + 1]
        x_t = sigma_t * noise + (1-sigma_t) * batch_img_1_embed
        #vis_image_only(vae, 0, i, x_t)
        traj.append(x_t)
    #exit()
    traj = torch.stack(traj, dim=0)
    traj= traj.squeeze(1)
    return traj

def compute_offline_prob(
    args,
    device,
    transformer,
    vae,
    img_1_embed,
    img_2_embed,
    caption,
    preference_embed, 
    reward_model,
    tokenizer,
    preprocess_val
    ):
    all_latents_1, all_latents_2 = [], []
    all_feature_1, all_feature_2 = [], []
    all_reward_1, all_reward_2 = [], []
    all_image_ids = []
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = args.w // SPATIAL_DOWNSAMPLE, args.h // SPATIAL_DOWNSAMPLE
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    B = img_1_embed.shape[0]
    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)
    for index, batch_idx in enumerate(batch_indices):
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        batch_img_1_embed = img_1_embed[batch_idx]
        batch_img_2_embed = img_2_embed[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        reward_1, img_feature_1 = vis_image(vae, index, 0, batch_img_1_embed,batch_caption, args.w,args.h, reward_model, tokenizer, preprocess_val, device)
        reward_2, img_feature_2 = vis_image(vae, index, 1, batch_img_2_embed,batch_caption, args.w,args.h, reward_model, tokenizer, preprocess_val, device)
        progress_bar = tqdm(range(0, args.sampling_steps), desc="Sampling Progress")
        traj_1 = flow_forward(batch_img_1_embed, sigma_schedule, progress_bar, vae)
        traj_2 = flow_forward(batch_img_2_embed, sigma_schedule, progress_bar, vae)
        all_latents_1.append(traj_1)
        all_latents_2.append(traj_2)
        all_feature_1.append(img_feature_1)
        all_feature_2.append(img_feature_2)
        all_reward_1.append(reward_1)
        all_reward_2.append(reward_2)
        all_image_ids.append(image_ids)
    all_latents_1 = torch.stack(all_latents_1, dim=0)
    all_latents_2 = torch.stack(all_latents_2, dim=0)
    all_feature_1 = torch.stack(all_feature_1, dim=0)
    all_feature_2 = torch.stack(all_feature_2, dim=0)
    all_reward_1 = torch.stack(all_reward_1, dim=0)
    all_reward_2 = torch.stack(all_reward_2, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)
    # print("all_latents_1 shape:", all_latents_1.shape)
    # print("all_latents_2 shape:", all_latents_2.shape)
    # print("all_feature_1 shape:", all_feature_1.shape)
    # print("all_feature_2 shape:", all_feature_2.shape)
    # print("all_reward_1 shape:", all_reward_1.shape)
    # print("all_reward_2 shape:", all_reward_2.shape)
    # exit()
    return all_latents_1, all_latents_2, all_feature_1, all_feature_2, all_reward_1, all_reward_2,sigma_schedule, all_image_ids


def vis_image(vae, index, ori, latents, batch_caption, w, h,reward_model, tokenizer,preprocess_val, device):
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    rank = int(os.environ["RANK"])
    
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = unpack_latents(latents, h, w, 8)
            latents = (latents / 0.3611) + 0.1159
            image = vae.decode(latents, return_dict=False)[0]
            decoded_image = image_processor.postprocess(
            image)
    decoded_image[0].save(f"./{args.save_image_path}/img_{rank}_{index}_{ori}.png")
    image_path = decoded_image[0]
    image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
    text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = reward_model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image)
    #print("image_features.shape", image_features.shape)
    image_features = image_features.squeeze(0)
    #print("image_features.shape", image_features.shape)
    #exit()
    return hps_score, image_features

def select_min_reward_samples(sample: Dict[str, Any], n: int) -> Dict[str, Any]:
    if 'rewards' not in sample:
        raise KeyError("样本字典中必须包含 'rewards' 键。")

    rewards = sample['rewards']

    rewards = rewards.squeeze()
    batch_size = rewards.size(0)

    if n <= 0:
        raise ValueError("分组步长 n 必须大于 0。")

    reward_groups = torch.split(rewards, n, dim=0)

    min_reward_global_indices = []
    
    current_global_start_index = 0
    
    for group_index, reward_group in enumerate(reward_groups):
        local_min_index = torch.argmin(reward_group)
        
        global_min_index = current_global_start_index + local_min_index
        
        min_reward_global_indices.append(global_min_index)
        
        current_global_start_index += reward_group.size(0)
        
    min_reward_indices_tensor = torch.tensor(min_reward_global_indices, 
                                             device=rewards.device)

    result_sample = {}
    
    for key, data in sample.items():
        if isinstance(data, torch.Tensor):
            result_sample[key] = data[min_reward_indices_tensor]
        else:
            # 对于非张量数据，直接复制
            result_sample[key] = data
    result_sample['preference'] = torch.full(
        size=result_sample['rewards'].shape, 
        fill_value=-1.0,
        dtype=result_sample['rewards'].dtype, # 使用最终 reward 的数据类型
        device=rewards.device,
    )
    return result_sample

def select_win_sample(samples_1: Dict[str, Any], samples_2: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较两个样本字典的 'preference' 键，并根据较大的 preference 值选择最终的键值对。
    同时，通过显式维度扩展，确保掩码能够正确广播到所有数据张量，解决 RuntimeError。
    """
    pref_1 = samples_1['preference']
    pref_2 = samples_2['preference']

    mask_1_wins = torch.gt(pref_1, pref_2)

    result_sample = {}

    for key in samples_1.keys():
        data_1 = samples_1[key]
        data_2 = samples_2[key]
        
        if not isinstance(data_1, torch.Tensor):
            result_sample[key] = data_1
            continue
        data_ndim = data_1.ndim
        mask_ndim = mask_1_wins.ndim
        dims_to_expand = data_ndim - mask_ndim
        broadcast_mask = mask_1_wins
        
        for _ in range(dims_to_expand):
            broadcast_mask = broadcast_mask[..., None]
        result_sample[key] = torch.where(broadcast_mask, data_1, data_2)
    return result_sample

def merge(sample_1: dict, sample_2: dict) -> dict:
    new_samples = {}
    for key in sample_1:
        t1 = sample_1[key]
        t2 = sample_2[key]
        N = t1.shape[0]
        combined = torch.empty(
            (2 * N, *t1.shape[1:]), 
            dtype=t1.dtype, 
            device=t1.device
        )
        combined[0::2], combined[1::2] = t1, t2
        new_samples[key] = combined
    return new_samples

def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    ema_handler,
    clip_model = None,
    preprocess_dgn5b = None,
    clip_tokenizer = None,
    transformer_reference = None
):
    #total_loss = 0.0
    optimizer.zero_grad()
    data_batch = next(loader)
    if args.strategy == "online":
        (
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            caption,
        ) = data_batch
    else:
        (
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            caption,
            img_1_embed, 
            img_2_embed, 
            preference_embed
        ) = data_batch
        encoder_hidden_states = encoder_hidden_states.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
        img_1_embed = img_1_embed.to(device)
        img_2_embed = img_2_embed.to(device)
        preference_embed[preference_embed == 0] = -1
        preference_embed = preference_embed.to(device)
        all_latents_1, all_latents_2, all_feature_1, all_feature_2, all_reward_1, all_reward_2, sigma_schedule, all_image_ids = compute_offline_prob(
            args,
            device,
            transformer,
            vae,
            img_1_embed,
            img_2_embed,
            caption,
            preference_embed,
            reward_model,
            tokenizer,
            preprocess_val,
            )
        all_log_probs = torch.zeros((all_latents_1.shape[0], sigma_schedule.shape[0] - 1), device=device)

        samples_1 = prepare_optimization_data(
                            args,
                            all_latents_1,
                            sigma_schedule,
                            all_log_probs,
                            all_image_ids,
                            all_reward_1,
                            text_ids,
                            encoder_hidden_states,
                            pooled_prompt_embeds,
                            all_predict_ori = all_predict_ori if args.consist else None,
                            all_clip_score=None,  
                            )

        samples_1['preference'] = preference_embed[:,0].unsqueeze(1)
        samples_2 = prepare_optimization_data(
                            args,
                            all_latents_2,
                            sigma_schedule,
                            all_log_probs,
                            all_image_ids,
                            all_reward_2,
                            text_ids,
                            encoder_hidden_states,
                            pooled_prompt_embeds,
                            all_predict_ori = all_predict_ori if args.consist else None,
                            all_clip_score=None,  # Optional, for use with hpsv2_clip
                            )
        samples_2['preference'] = preference_embed[:,1].unsqueeze(1)

        samples_offline = merge(samples_1, samples_2)
        if args.strategy == "offline":
            opt_ret = execute_optimization_step(
                args,
                samples_offline,
                transformer,
                optimizer,
                lr_scheduler,
                sigma_schedule,
                max_grad_norm,
                vae = vae,
                transformer_reference = transformer_reference,
            )
            return opt_ret
        else:
            samples_offline = select_win_sample(samples_1, samples_2)

    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)
        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")
    if args.use_hpsv2_clip:
        reward, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_clip_score = sample_reference_model(
                args,
                device, 
                transformer,
                vae,
                encoder_hidden_states, 
                pooled_prompt_embeds, 
                text_ids,
                reward_model,
                tokenizer,
                caption,
                preprocess_val,
                clip_model,
                preprocess_dgn5b,
                clip_tokenizer,
            )
    else:
        ret = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
        )
        if args.consist:
            reward, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_predict_ori = ret
        else:
            reward, all_latents, all_log_probs, sigma_schedule, all_image_ids = ret
        #print("!!!")
        #exit()
    # ----------------------------------------------------

    samples_online = prepare_optimization_data(
                            args,
                            all_latents,
                            sigma_schedule,
                            all_log_probs,
                            all_image_ids,
                            reward,
                            text_ids,
                            encoder_hidden_states,
                            pooled_prompt_embeds,
                            all_predict_ori = all_predict_ori if args.consist else None,
                            all_clip_score= all_clip_score if args.use_hpsv2_clip else None
                            )
    samples_online['rewards'] = samples_online['rewards'].unsqueeze(1)
    if args.strategy != "online":
        sample_online_mini = select_min_reward_samples(samples_online, args.num_generations)
        samples = merge(samples_offline, sample_online_mini)
    #for key, value in samples_online.items():
    #    if isinstance(value, torch.Tensor):
    #        print(f"samples {key}: {value.shape}")
    #exit()
    opt_ret = execute_optimization_step(
        args,
        samples_online,
        transformer,
        optimizer,
        lr_scheduler,
        sigma_schedule,
        max_grad_norm,
        vae = vae,
        transformer_reference = transformer_reference
    )

    return opt_ret
    #if args.use_hpsv2_clip:
    #    return total_loss, grad_norm.item(), sum(reward_avg) / len(reward_avg), sum(reward_std) / len(reward_std), sum(clip_score_avg) / len(clip_score_avg), sum(clip_score_std) / len(clip_score_std)
    #else:
    #    return total_loss, grad_norm.item(), sum(reward_avg) / len(reward_avg), sum(reward_std) / len(reward_std)

def prepare_optimization_data(
    args,
    all_latents,
    sigma_schedule,
    all_log_probs,
    all_image_ids,
    reward,
    text_ids,
    encoder_hidden_states,
    pooled_prompt_embeds,
    all_predict_ori=None,
    all_clip_score=None,  # Optional, for use with hpsv2_clip
):
    """
    Prepares the data batch for a single optimization step.

    This function sets up the 'samples' dictionary with latents, rewards,
    and other necessary metadata. It also handles diversity sampling if enabled.
    """
    batch_size = all_latents.shape[0]
    number_step = all_latents.shape[1] - 1
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:number_step]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    if args.consist:
        samples = {
            "timesteps": timesteps.detach().clone()[:, :-1],
            "latents": all_latents[:, :-1][:, :-1],  # Latent before timestep t
            "next_latents": all_latents[:, 1:][:, :-1],  # Latent after timestep t
            "log_probs": all_log_probs[:, :-1],
            "rewards": reward.to(torch.float32),
            "image_ids": all_image_ids,
            "text_ids": text_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "predict_ori":all_predict_ori[:, 1:][:, :-1], # predicted clean sample using the next-step sample
        }
        #print(samples['predict_ori'].shape)
        #print(samples['latents'].shape)
        #exit()
    else:
        samples = {
            "timesteps": timesteps.detach().clone()[:, :-1],
            "latents": all_latents[:, :-1][:, :-1],  # Latent before timestep t
            "next_latents": all_latents[:, 1:][:, :-1],  # Latent after timestep t
            "log_probs": all_log_probs[:, :-1],
            "rewards": reward.to(torch.float32),
            "image_ids": all_image_ids,
            "text_ids": text_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    if args.use_hpsv2_clip and all_clip_score is not None:
        samples["clip_scores"] = all_clip_score.to(torch.float32)
    return samples

def _calculate_advantages(args, samples, gathered_reward, num_generations):
    """
    Helper function to calculate advantages for a given set of samples.
    (辅助函数，为给定样本集计算优势函数。)
    """
    reward_avg, reward_std = [], []
    clip_score_avg, clip_score_std = [], []
    
    if args.use_group:
        # print("len(samples[rewards])", len(samples["rewards"]))
        # print("num_generations", num_generations)
        n = len(samples["rewards"]) // num_generations
        advantages = torch.zeros_like(samples["rewards"])
        #preference = torch.zeros_like(samples["rewards"])
        for i in range(n):
            start_idx = i * num_generations
            end_idx = (i + 1) * num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]

            if args.use_hpsv2_clip:
                group_clip_scores = samples["clip_scores"][start_idx:end_idx]
                clip_mean, clip_std = group_clip_scores.mean(), group_clip_scores.std() + 1e-8
                clip_score_avg.append(clip_mean)
                clip_score_std.append(clip_std)
                group_rewards = group_rewards * 0.7 + group_clip_scores * 1.4

            group_mean, group_std = group_rewards.mean(), group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            reward_avg.append(group_mean)
            reward_std.append(group_std)
        if args.strategy == "online":
            samples["preference"] = torch.where(advantages > 0, torch.tensor(1.0), torch.tensor(-1.0))
        samples["advantages"] = advantages
        # print(samples["preference"])
        # print(samples["advantages"])
        # exit()
    else:
        ValueError("No implementation")

    return samples, reward_avg, reward_std, clip_score_avg, clip_score_std

def _compute_new_probs(args, sample, perms, i, t_idx, transformer, sigma_schedule, train_timesteps, vae):
    """
    Helper function to compute the loss for one timestep and perform backpropagation.
    (辅助函数，计算单个时间步的损失并执行反向传播。)
    """
    ret = grpo_one_step(
        args,
        sample["latents"][:, t_idx],
        sample["next_latents"][:, t_idx],
        sample["encoder_hidden_states"],
        sample["pooled_prompt_embeds"],
        sample["text_ids"],
        sample["image_ids"],
        transformer,
        sample["timesteps"][:, t_idx],
        perms[i][t_idx],
        sigma_schedule,
    )
    if args.consist:
        new_log_probs, new_predict_ori = ret
    else:
        new_log_probs = ret
    return ret

def execute_optimization_step(
    args,
    samples,
    transformer,
    optimizer,
    lr_scheduler,
    sigma_schedule,
    max_grad_norm,
    vae=None,
    transformer_reference = None,
):
    """
    Executes the optimization step using prepared sample data.
    If `samples_2` is provided, it performs optimization on both sample sets.
    Otherwise, it only processes `samples`.
    """
    total_loss = 0.0
    total_loss_consist = 0.0
    batch_size = samples["latents"].shape[0]
    device = samples["latents"].device

    # --- Calculate advantages ---
    gathered_reward = gather_tensor(samples["rewards"])
    # if args.strategy == 'sopo':
    samples, reward_avg, reward_std, clip_avg, clip_std = _calculate_advantages(args, samples, gathered_reward, 2)

    def shuffle_timesteps(samples, batch_size, device):
        """
        Shuffles timesteps and ensures the resulting tensors are contiguous.
        """
        perms = torch.stack([torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)]).to(device)
        keys = ["timesteps", "latents", "next_latents", "log_probs", "predict_ori"] if args.consist else ["timesteps", "latents", "next_latents", "log_probs"]
        for key in keys:
            #print(key)
            # The advanced indexing operation below creates a non-contiguous tensor.
            samples[key] = samples[key][torch.arange(batch_size).to(device)[:, None], perms]
            # We explicitly call .contiguous() to ensure the tensor has a contiguous memory layout.
            samples[key] = samples[key].contiguous()
        return samples, perms

    # Shuffle only `samples` if `samples_2` is not provided
    samples, perms = shuffle_timesteps(samples, batch_size, device)
    
    # --- Prepare batched data ---
    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    samples_batched_list = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]


    # --- Training Loop ---
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)

    #accumulation_counter = 0
    grad_norm = torch.tensor(0.0, device=device)  # Initialize grad_norm

    # Iterate over the longest list of batches to ensure all data is processed
    max_len = len(samples_batched_list)

    for k in range(0, max_len, 2):
        # Process a batch from the first set of samples
        if k < len(samples_batched_list):
            loss = 0.0
            sample_k_1 = samples_batched_list[k]
            #print("sample_k_1[preference]", sample_k_1["preference"])
            sample_k_1 = {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v for k, v in sample_k_1.items()}
            
            sample_k_2 = samples_batched_list[k+1]
            #print("sample_k_2[preference]", sample_k_2["preference"])
            sample_k_2 = {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v for k, v in sample_k_2.items()}

            for t_idx in range(train_timesteps):
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                ret_1 = _compute_new_probs(args, sample_k_1, perms, k, t_idx, transformer, sigma_schedule, train_timesteps, vae)
                ret_2 = _compute_new_probs(args, sample_k_2, perms, k + 1, t_idx, transformer, sigma_schedule, train_timesteps, vae)
                if args.consist:
                    probs_t_k1, new_predict_ori_1  = ret_1
                    probs_t_k2, new_predict_ori_2  = ret_2

                    old_predict_ori_1, old_predict_ori_2 = sample_k_1["predict_ori"][:, t_idx], sample_k_2["predict_ori"][:, t_idx]
                    loss_consist = torch.norm(old_predict_ori_1 - new_predict_ori_1)
                    loss_consist += torch.norm(old_predict_ori_2 - new_predict_ori_2)
                    loss_consist = args.consist_weight * torch.mean(loss_consist)
                    avg_loss_consist = loss_consist.clone()
                    dist.all_reduce(avg_loss_consist, op=dist.ReduceOp.AVG)
                    if (sample_k_1["timesteps"][:, t_idx] <= 600).any() and torch.rand(1).item() < 0.2:
                        total_loss_consist += avg_loss_consist.item() / max_len  # Normalize loss contribution
                else:
                    probs_t_k1  = ret_1
                    probs_t_k2  = ret_2

                if args.use_reference:
                    #transformer.disable_adapters()
                    probs_t_k1_old = _compute_new_probs(args, sample_k_1, perms, k, t_idx, transformer_reference, sigma_schedule, train_timesteps, vae)
                    probs_t_k2_old = _compute_new_probs(args, sample_k_2, perms, k + 1, t_idx, transformer_reference, sigma_schedule, train_timesteps, vae)
                    #transformer.enable_adapters()
                else:
                    probs_t_k1_old = sample_k_1["log_probs"][:, t_idx]
                    probs_t_k2_old = sample_k_2["log_probs"][:, t_idx]

                log_ratio_t_k1 = probs_t_k1 - probs_t_k1_old
                log_ratio_t_k2 = probs_t_k2 - probs_t_k2_old

                # prob_gap = sample_k_1["preference"] * log_ratio_t_k1 + sample_k_2["preference"] * log_ratio_t_k2
                # loss = - 1 / (1 + torch.exp(- args.dpo_beta * prob_gap))
                # loss /= (args.gradient_accumulation_steps * train_timesteps)

                if args.use_sft:
                    # SFT mode: maximize log probability of positive samples
                    loss = -probs_t_k1  # Negative because we want to maximize
                    loss /= (args.gradient_accumulation_steps * train_timesteps)
                    prob_gap = probs_t_k1
                    loss = loss
                else:
                    # DPO mode: standard DPO loss
                    prob_gap = sample_k_1["preference"] * log_ratio_t_k1 + sample_k_2["preference"] * log_ratio_t_k2
                    loss = - 1 / (1 + torch.exp(- args.dpo_beta * prob_gap))
                    loss /= (args.gradient_accumulation_steps * train_timesteps)
                
                # prob_gap = sample_k_1["preference"] * log_ratio_t_k1 + sample_k_2["preference"] * log_ratio_t_k2
                # loss = - 1 / (1 + torch.exp(- args.dpo_beta * prob_gap))
                # loss /= (args.gradient_accumulation_steps * train_timesteps)
                #loss = -log_ratio_t_k1
                #loss = probs_t_k1
                # if dist.get_rank() % 8 == 0:
                #     print("probs_t_k1_old 是否可微 (requires_grad):", probs_t_k1_old.requires_grad)
                #     print("probs_t_k2 是否可微 (requires_grad):", probs_t_k2.requires_grad)
                #loss = prob_gap

                loss.backward()
                #exit()
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                total_loss += avg_loss.item() / max_len  # Normalize loss contribution

        if dist.get_rank() % 8 == 0:
            loss_consist_print = loss_consist.item() if args.consist else 0.0
            if k < len(samples_batched_list):
                print(f"[Set 1, Batch {k}] reward: {sample_k_1['rewards'].item():.4f}, preference: {sample_k_1['preference'].item():.4f}, prob_gap: {prob_gap.mean().item():.4f}, loss: {loss.item():.4f}, loss_consist: {loss_consist_print:.4f}, log_ratio_t_k1: {log_ratio_t_k1.item():.4f}")
                print(f"[Set 1, Batch {k+1}] reward: {sample_k_2['rewards'].item():.4f}, preference: {sample_k_2['preference'].item():.4f}, prob_gap: {prob_gap.mean().item():.4f}, loss: {loss.item():.4f}, loss_consist: {loss_consist_print:.4f}, log_ratio_t_k2: {log_ratio_t_k2.item():.4f}")
            else:
                print(f"[Set 2, Batch {k}] reward: {sample2['rewards'].item():.4f}, preference: {sample2['preference'].item():.4f}, ratio: {ratio_2.mean().item():.4f}, loss: {loss_2.item():.4f}, loss_consist: {loss_consist_print:.4f}")

        if (k+2) % args.gradient_accumulation_steps == 0:
            if dist.get_rank() % 8 == 0:
                print("opt!!!")
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    dist.barrier()

    # --- Aggregate and Return Metrics ---
    final_reward_avg = (sum(reward_avg) / len(reward_avg))
    final_reward_std = (sum(reward_std) / len(reward_std))

    if args.consist:
        return total_loss, grad_norm.item(), final_reward_avg.item(), final_reward_std.item(), total_loss_consist
    return total_loss, grad_norm.item(), final_reward_avg.item(), final_reward_std.item()

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    preprocess_val = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './/pretrained_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        # cp = "./hps_ckpt/HPS_v2.1_compressed.pt"
        cp = ".//pretrained_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()
    if args.use_hpsv2_clip:
        from open_clip import create_model_from_pretrained, get_tokenizer
        clip_tokenizer = get_tokenizer('ViT-H-14')
        clip_model, preprocess_dgn5b = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
        clip_model.to(device).eval()
        #clip_model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
        #clip_model = clip_model.half().to(device)
        #local_model_path = "./pretrained_ckpt/apple_clip"
        #clip_model, preprocess = create_model_from_pretrained(local_model_path, device=device)
        #clip_tokenizer = get_tokenizer('ViT-H-14')
        # processor = CLIPProcessor.from_pretrained("./pretrained_ckpt/apple_clip")
        # model_pretrained_name_or_path = "./pretrained_ckpt/apple_clip"
        # reward_model = CLIPModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    if args.use_pickscore:
        from transformers import AutoProcessor, AutoModel
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        reward_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )
    transformer_reference = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32 # 保持与训练模型主权重相同的 dtype
    )

    # 2. 将参考模型设置为不可训练（必须）。
    transformer_reference.requires_grad_(False)
    transformer_reference.to(device)
    # 3. 将参考模型设置为评估模式（推理模式，必须）。
    transformer_reference.eval()

    from diffusers import FluxPipeline
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft
    pipe = FluxPipeline
    transformer.requires_grad_(False)
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    if args.resume_from_lora_checkpoint:
        lora_state_dict = pipe.lora_state_dict(
            args.resume_from_lora_checkpoint)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(
            transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer,
                                                      transformer_state_dict,
                                                      adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys",
                                      None)
            if unexpected_keys:
                main_print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. ")

    # transformer.disable_adapters()
    # transformer.enable_adapters()
    # transformer.disable_adapters()

    #print(args.master_weight_type)
    #exit()
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        True,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    transformer.config.lora_rank = args.lora_rank
    transformer.config.lora_alpha = args.lora_alpha
    transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    transformer._no_split_modules = [
        no_split_module.__name__ for no_split_module in no_split_modules
    ]
    fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)

    transformer = FSDP(transformer, **fsdp_kwargs,)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        #print("resume_from_lora_checkpoint?", resume_from_lora_checkpoint)
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )
        #print("init_steps", init_steps)
        #exit()
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    if args.strategy != "online":
        from fastvideo.dataset.latent_flux_rl_datasets import OfflineLatentDataset as LatentDataset
        from fastvideo.dataset.latent_flux_rl_datasets import offline_latent_collate_function as latent_collate_function
        args.data_json_path = args.offline_path
    else:
        from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )


    #vae.enable_tiling()

    if rank <= 0:
        #project = "debug"
        project = args.project
        #wandb.init(project=project, config=args)
        swanlab.init(project=project, workspace="xiaofengtan",config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        transformer = resume_training_w_o_optimizer(transformer, args.resume_from_checkpoint)
        init_steps = int(args.resume_steps)
        print(f"Resuming training from step {init_steps}")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )
    if args.strategy != "online":
        #loader = train_dataloader
        loader = iter(train_dataloader)

    step_times = deque(maxlen=100)
            
    num_steps = args.max_train_steps + 1

    # rank 0 上生成 skip_rate 列表
    
    if rank == 0:
        random.seed(42)
        skip_rates = [
            args.skip_rate_range + (1 - args.skip_rate_range) * random.random()
            if args.skip_rate_range != 1.0 else 1.0
            for _ in range(num_steps)
        ]
        skip_rates_tensor = torch.tensor(skip_rates, dtype=torch.float32, device=device)
    else:
        skip_rates_tensor = torch.empty(num_steps, dtype=torch.float32, device=device)
    # 广播到所有 GPU
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.broadcast(skip_rates_tensor, src=0)
    # 转回列表
    skip_rates = skip_rates_tensor.tolist()
    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        
        for step in range(0, args.max_train_steps+1):
        #for step in range(init_steps+1, args.max_train_steps+1):
            if step < init_steps:
                _ = next(loader)
                print(f"Skipping step {step}")
                continue
            #print(step)
            
            args.skip_rate = skip_rates[step]
            print(f"step {step} skip_rate: {args.skip_rate}")
            print(skip_rates)
            print(len(skip_rates))
            #if args.skip_rate_range != 1.0:
            #    args.skip_rate = args.skip_rate_range + (1 - args.skip_rate_range) * random.uniform(0, 1)
            if args.mix:
                print("mix!!!")
                print("--mix_rate", args.mix_rate)
                print("--mix_period", args.mix_period)
                if (step % args.mix_period) / args.mix_period  < (1 - args.mix_rate):
                    print("same-step", step)
                    args.init_same_noise = True
                else:
                    print("diff-step", step)
                    args.init_same_noise = False
            start_time = time.time()

            if step % args.checkpointing_steps == 0:
                save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                                        step, pipe, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))
                dist.barrier()
            if args.use_hpsv2_clip:
                loss, grad_norm, reward_avg, reward_std, clip_avg, clip_std  = train_one_step(
                    args,
                    device, 
                    transformer,
                    vae,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    noise_scheduler,
                    args.max_grad_norm,
                    preprocess_val,
                    ema_handler,
                    clip_model,
                    preprocess_dgn5b,
                    clip_tokenizer
                )
            else:
                ret_train = train_one_step(
                    args,
                    device, 
                    transformer,
                    vae,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    noise_scheduler,
                    args.max_grad_norm,
                    preprocess_val,
                    ema_handler,
                    transformer_reference = transformer_reference if args.use_reference else None ,
                )
                if args.consist:
                    loss, grad_norm, reward_avg, reward_std, loss_consist = ret_train
                else:
                    loss, grad_norm, reward_avg, reward_std = ret_train
            if args.use_ema and ema_handler:
                ema_handler.update(transformer)
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                if args.use_hpsv2_clip:
                    swanlab.log(
                        {
                            "train_loss": loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step_time": step_time,
                            "avg_step_time": avg_step_time,
                            "grad_norm": grad_norm,
                            "reward_avg": reward_avg, 
                            "reward_std": reward_std,
                            "skip_rate": args.skip_rate,
                            "clip_avg": clip_avg,
                            "clip_std": clip_std,
                            "init_same_noise": args.init_same_noise
                        },
                        step=step,
                    )
                else:
                    if args.consist:
                        swanlab.log(
                            {
                                "train_loss": loss,
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "step_time": step_time,
                                "avg_step_time": avg_step_time,
                                "grad_norm": grad_norm,
                                "reward_avg": reward_avg, 
                                "reward_std": reward_std,
                                "skip_rate": args.skip_rate,
                                "init_same_noise": args.init_same_noise,
                                "loss_consist": loss_consist
                            },
                            step=step,
                        )
                    else:
                        swanlab.log(
                            {
                                "train_loss": loss,
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "step_time": step_time,
                                "avg_step_time": avg_step_time,
                                "grad_norm": grad_norm,
                                "reward_avg": reward_avg, 
                                "reward_std": reward_std,
                                "skip_rate": args.skip_rate,
                                "init_same_noise": args.init_same_noise
                            },
                            step=step,
                        )


    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # resume_from_lora_checkpoint
    
    parser.add_argument(
        "--resume_steps",
        type=str,
        default=None,
        help=(
            "resume_steps"
        ),
    )
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_hpsv2_clip",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--diversity",
        action="store_true",
        default=False,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--remain_div",
        action="store_true",
        default=False,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--diversity_step",
        type=int,
        default=10,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--diversity_rate",
        type=float,
        default=2.0,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--select_by",
        type=str,
        default="feature",
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--select_method",
        type=str,
        default="kmeans",
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        default=False,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--mix_period",
        type=int,
        default=40,
        #period help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--mix_rate",
        type=float,
        default=0.25,
        #period help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--consist",
        action="store_true",
        default=False,
        #help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--consist_weight",
        type = float,
        default=0.001,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--dpo_beta",
        type = float,
        default=1,
        help="DPO beta",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    )
    
    parser.add_argument(
        "--skip_rate",
        type = float,
        default=1.0,
        help="clipping advantage",
    )
    
    parser.add_argument(
        "--skip_rate_range",
        type = float,
        default=1.0,
        help="clipping advantage",
    )

    parser.add_argument(
        "--save_image_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter. "
    )
    parser.add_argument(
        "--use_reference",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="online",
        help="online, offline, or, semi-online",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="debug",
        help="project name",
    )
    parser.add_argument(
        "--use_sft",
        action="store_true",
        default=False,
        help="Use SFT (Supervised Fine-Tuning) on positive samples from offline DPO",
    )
    parser.add_argument("--offline_path", type=str, default="HPS_dataset/rl_embeddings/videos2caption.json")
    args = parser.parse_args()
    main(args)
