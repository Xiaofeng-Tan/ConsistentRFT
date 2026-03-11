import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available
from torch.utils.data import Dataset, DistributedSampler
from safetensors.torch import load_file
import argparse
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import copy
import json


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False



class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main(args):
    rank, local_rank, world_size = distributed_setup()
    if rank == 0:
        for key, value in vars(args).items():
            print(f"{key}: {value}")
    # dataset
    dataset = PromptDataset(args.prompts_file)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(
        args.flux_baseline_model_dir,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")

    #print(pipe.transformer)
    #exit()
    # Load the model checkpoint
    if not args.baseline:        
        if args.lora_rank == 0:
            model_state_dict = load_file(args.model_path)
            pipe.transformer.load_state_dict(model_state_dict, strict=True)
            pipe.to("cuda")
        else:
            from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
            from diffusers.utils import convert_unet_state_dict_to_peft
            #pipe = FluxPipeline
            pipe.transformer.requires_grad_(False)
            transformer_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            pipe.transformer.add_adapter(transformer_lora_config)


            lora_state_dict = pipe.lora_state_dict(
                args.model_path)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(
                transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(pipe.transformer,
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
        

    # inference
    meta_data = []
    for idx in sampler:
        prompt = dataset[idx]
        # if idx >= 10:
        #     break
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(args.seed + idx + rank*1000)

            save_path = output_dir / f"{idx}.jpg"
            if not save_path.exists():
                if args.mix_sampling_steps > 0:
                    image = pipe(
                        prompt,
                        guidance_scale=3.5,
                        height=1024,
                        width=1024,
                        num_inference_steps=args.total_sampling_steps,
                        max_sequence_length=512,
                        generator=generator,
                        mix_sampling_steps=args.mix_sampling_steps
                    ).images[0]
                else:
                    image = pipe(
                        prompt,
                        guidance_scale=3.5,
                        height=1024,
                        width=1024,
                        num_inference_steps=args.total_sampling_steps,
                        max_sequence_length=512,
                        generator=generator,

                    ).images[0]
                image.save(save_path)

            meta_data.append({
                "image": str(save_path),
                "prompt": prompt,
            })
            print(f"[Rank {rank}] Generated: {save_path.name} for prompt: {prompt[:20]}...")
        except Exception as e:
            raise(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")
    
    # gather metadata from all ranks
    all_meta_data = [None] * world_size
    #dist.all_gather_object(all_meta_data, meta_data)
    def all_gather_object_gloo(obj):
        world_size = dist.get_world_size()
        out_list = [None for _ in range(world_size)]
        try:
            dist.all_gather_object(out_list, obj)  # 先试默认(可能是NCCL)
            return out_list
        except Exception:
            # fallback 到 Gloo 子组
            pg = dist.new_group(backend="gloo")
            dist.all_gather_object(out_list, obj, group=pg)
            # 可选：销毁子组
            dist.destroy_process_group(pg)
            return out_list
    all_meta_data = all_gather_object_gloo(meta_data)
    meta_data_results = []
    for rank_meta in all_meta_data:
        if rank_meta is not None:
            meta_data_results.extend(rank_meta)

    
    with open(args.output_json, "w") as f:
        json.dump(meta_data_results, f, indent=4)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Inference for MixGRPO")
    parser.add_argument("--model_path", type=str,
                        help="Path to the MixGRPO model checkpoint")
    parser.add_argument("--prompts_file", type=str, default="./data/prompts_test.txt",
                        help="Path to the file containing prompts")
    parser.add_argument("--output_dir", type=str, default="./output_flux",
                        help="Directory to save generated images")
    parser.add_argument("--output_json", type=str, default="output_flux.json",
                        help="Path to save the output JSON file with metadata")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random number generation")
    parser.add_argument("--baseline", action='store_true', default=False,
                        help="Use baseline model settings")
    parser.add_argument("--mix_sampling_steps", type=int, default=-1,
                        help="Number of sampling steps of the MixGRPO model")
    parser.add_argument("--total_sampling_steps", type=int, default=50,
                        help="Total number of sampling steps")
    parser.add_argument("--flux_baseline_model_dir", type=str, default="./data/flux",)
    parser.add_argument("--lora_alpha", type=int, default=0,
                        help="")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="")
    
    args = parser.parse_args()

    main(args)
