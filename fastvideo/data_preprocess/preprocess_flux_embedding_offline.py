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


import argparse
import torch
from accelerate.logging import get_logger
import cv2  
import json
import os
import torch.distributed as dist
from pathlib import Path  
from PIL import Image, UnidentifiedImageError
import numpy as np

logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from diffusers import FluxPipeline, AutoencoderKL
import json

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def encode_image(image_path, vae, preprocess_func, device):
    """
    处理并编码图像，返回处理后的潜在向量。

    参数:
    - image_path (str): 图像的文件路径。
    - vae (VAE): VAE模型。
    - preprocess_func (function): 图像预处理函数。
    - latent_shape (tuple): 潜在空间的形状，(height, width)。
    - device (torch.device): 目标设备。

    返回:
    - latent_z (Tensor): 编码后的潜在向量。
    """
    if not os.path.exists(image_path):
        print(f"警告: 图像文件 {image_path} 不存在，跳过处理。")
        return None  # 文件不存在时，返回 None
    # 打开图像
    # image = Image.open(image_path)
    try:
        # 打开图像
        image = Image.open(image_path)
    except UnidentifiedImageError:
        print(f"警告: 图像文件 {image_path} 无法识别，跳过处理。")
        return None  # 如果图像无法识别，跳过

    if image.mode != 'RGB':
        print(f"跳过图像 {image_path}，因为它不是三通道图像（当前模式: {image.mode}）")
        return None  # 如果不是三通道，返回 None

    # 预处理并调整图像大小
    processed_image = preprocess_func(image, device, 720, torch.bfloat16)

    # 编码图像，获得潜在表示
    latent_z = vae.encode(processed_image)
    latent_z = (latent_z.latent_dist.sample().detach() - 0.1159) * 0.3611

    # 包装潜在向量
    # latent_z_packed = pack_latents(latent_z, *latent_shape)
    latent_z_packed = pack_latents(latent_z, 1, 16, 90, 90)
    # 打印信息
    # print(f"处理后的图像形状 (H, W, C): {processed_image.shape}")
    # print(f"潜在向量 z 形状: {latent_z.shape}")
    # print(f"包装后的潜在向量 z 形状: {latent_z_packed.shape}")

    return latent_z_packed

class T5dataset_image_text(Dataset):
    def __init__(
        self, json_path, number_pair,
    ):
        self.json_path = json_path
        self.number_pair = number_pair
        #self.vae_debug = vae_debug
        # with open(self.txt_path, "r", encoding="utf-8") as f:
        #     self.train_dataset = [
        # line for line in f.read().splitlines() if not contains_chinese(line)
        # ][:50000]
        self.train_dataset = json.load(open(json_path,"r"))[0:self.number_pair]

    def __getitem__(self, idx):
        #import pdb;pdb.set_trace()
        caption = self.train_dataset[idx]['prompt']
        image_path = self.train_dataset[idx]['image_path']
        #print(image_path)
        #exit()
        human_preference = self.train_dataset[idx]['human_preference']
        #human_preference = [t.item() for t in self.train_dataset[idx]['human_preference']]

        filename = str(idx)
        #length = self.train_dataset[idx]["length"]
        # if self.vae_debug:
        #     latents = torch.load(
        #         os.path.join(
        #             args.output_dir, "latent", self.train_dataset[idx]["latent_path"]
        #         ),
        #         map_location="cpu",
        #     )
        # else:
        #     latents = []

        return dict(caption=caption, image_path=image_path, human_preference=human_preference, filename=filename)

    def __len__(self):
        return len(self.train_dataset)

# 目标图片尺寸
TARGET_SIZE = 720 
TARGET_DTYPE = torch.bfloat16 # 使用 BFloat16 解决 Runtime Error

def preprocess_image_and_resize(image: Image.Image, device: torch.device, target_size: int, target_dtype: torch.dtype):
    """
    强制将 PIL 图片缩放（拉伸）到目标尺寸，并进行 VAE 所需的标准化和格式转换。
    """
    
    # **核心操作：强制缩放（拉伸）到 target_size x target_size**
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        #print(f"Image resized from original {image.size} to {target_size}x{target_size}.")

    # PIL Image -> NumPy -> PyTorch Tensor
    img_np = np.array(image).astype(np.float32) / 255.0 # 归一化到 [0, 1]
    
    # 格式转换 [H, W, C] -> [C, H, W]
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

    # 标准化到 [-1, 1] 范围
    img_tensor = 2.0 * img_tensor - 1.0 
    
    # **关键：转换为 BFloat16 精度**
    return img_tensor.to(device).to(target_dtype)

def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embeds"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "img_1_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "img_2_embed"), exist_ok=True)
    #os.makedirs(os.path.join(args.output_dir, "human_preference"), exist_ok=True)

    json_path = args.json_path
    train_dataset = T5dataset_image_text(json_path, args.number_pair)
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    #pipe = FluxPipeline.from_pretrained(
    #    args.model_path,
    #    torch_dtype=torch.bfloat16,
    #    use_safetensors=True
    #).to(device)
    # vae = AutoencoderKL.from_pretrained(
    #     args.model_path,
    #     subfolder="vae",
    #     torch_dtype = torch.bfloat16,
    # ).to(device)
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    # encode_image
    #exit()
    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        try:
            with torch.inference_mode():
                if args.vae_debug:
                    latents = data["latents"]
                for idx, video_name in enumerate(data["filename"]):
                    # 构建保存路径
                    latent_z_1_path = os.path.join(args.output_dir, "img_1_embed", video_name + ".pt")
                    latent_z_2_path = os.path.join(args.output_dir, "img_2_embed", video_name + ".pt")
                    preference_path = os.path.join(args.output_dir, "human_preference", video_name + ".pt")
                    prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                    pooled_prompt_embeds_path = os.path.join(args.output_dir, "pooled_prompt_embeds", video_name + ".pt")
                    text_ids_path = os.path.join(args.output_dir, "text_ids", video_name + ".pt")

                    #如果文件已经存在，直接跳过
                    if (os.path.exists(latent_z_1_path) and os.path.exists(latent_z_2_path) 
                        and os.path.exists(preference_path) and os.path.exists(prompt_embed_path)
                        and os.path.exists(pooled_prompt_embeds_path) and os.path.exists(text_ids_path)):
                        # json 信息
                        human_preference = [t.item() for t in data["human_preference"]]
                        item = {
                            "prompt_embed_path": video_name + ".pt",
                            "text_ids": video_name + ".pt",
                            "pooled_prompt_embeds_path": video_name + ".pt",
                            "caption": data["caption"][idx],
                            "human_preference": human_preference,
                            "latent_z_1_path": video_name + ".pt",
                            "latent_z_2_path": video_name + ".pt",
                        }
                        json_data.append(item)
                        continue

                    # ---------------- 只处理需要的样本 ----------------
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=data["caption"], prompt_2=data["caption"]
                    )
                    image_path_1 = f"{args.img_dir}/{data['image_path'][0][0]}"
                    image_path_2 = f"{args.img_dir}/{data['image_path'][1][0]}"

                    latent_z_1 = encode_image(image_path_1, pipe.vae, preprocess_image_and_resize, device)
                    latent_z_2 = encode_image(image_path_2, pipe.vae, preprocess_image_and_resize, device)
                    if latent_z_1 == None or latent_z_2 ==None:
                        print(image_path_1, image_path_1)
                        continue
                    # 保存结果
                    torch.save(latent_z_1, latent_z_1_path)
                    torch.save(latent_z_2, latent_z_2_path)
                    torch.save(data["caption"], preference_path)
                    torch.save(prompt_embeds[idx], prompt_embed_path)
                    torch.save(pooled_prompt_embeds[idx], pooled_prompt_embeds_path)
                    torch.save(text_ids[idx], text_ids_path)
                    human_preference = [t.item() for t in data["human_preference"]]
                    item = {
                        "prompt_embed_path": video_name + ".pt",
                        "text_ids": video_name + ".pt",
                        "pooled_prompt_embeds_path": video_name + ".pt",
                        "caption": data["caption"][idx],
                        "human_preference": human_preference,
                        "latent_z_1_path": video_name + ".pt",
                        "latent_z_2_path": video_name + ".pt",
                    }
                    json_data.append(item)

        except Exception as e:
            print(f"Rank {local_rank} Error: {repr(e)}")
            dist.barrier()
            raise
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    #parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
    parser.add_argument("--json_path", type=str, default="./empty.txt")
    parser.add_argument("--number_pair", type=int, default=50)
    parser.add_argument("--img_dir", type=str, default="50")
    args = parser.parse_args()
    main(args)